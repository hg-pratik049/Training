# -*- coding: utf-8 -*-
import os
from datetime import date, timedelta
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# ======================================================
# Streamlit Page Config (set once)
# ======================================================
st.set_page_config(
    page_title="MySQL Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Retail Analytics – MySQL + Streamlit")

# ======================================================
# Utilities
# ======================================================
def download_button(df: pd.DataFrame, label: str = "⬇️ Download CSV", filename: str = "data.csv"):
    if df is None or df.empty:
        st.info("No data to download.")
        return
    csv = df.to_csv(index=False)
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")

@st.cache_resource(show_spinner=False)
def get_engine():
    """
    Returns a SQLAlchemy engine.
    1) Tries Streamlit secrets: st.secrets['mysql']
    2) If missing, shows sidebar inputs for manual credentials.
    """
    cfg = None
    try:
        cfg = st.secrets["mysql"]
    except Exception:
        st.sidebar.warning("No secrets.toml found. Enter MySQL credentials below.")
        host = st.sidebar.text_input("Host", value="localhost")
        port = st.sidebar.number_input("Port", value=3306, step=1)
        database = st.sidebar.text_input("Database")
        user = st.sidebar.text_input("User")
        password = st.sidebar.text_input("Password", type="password")
        if not all([host, port, database, user, password]):
            st.stop()  # Wait until user enters all values
        cfg = {"host": host, "port": int(port), "database": database, "user": user, "password": password}

    conn_str = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg.get('port',3306)}/{cfg['database']}?charset=utf8mb4"
    )
    engine = create_engine(conn_str, pool_pre_ping=True)
    return engine

def run_query(sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Runs a SQL query using pandas + SQLAlchemy. Returns a DataFrame.
    Accepts either a plain SQL string or a SQL string with :named params.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df

def exec_sql(sql: str, params: Optional[Dict[str, Any]] = None):
    """
    Execute a non-SELECT SQL (DDL/DML).
    """
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def default_year_range(year: int) -> Tuple[date, date]:
    start = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    return start, end

def add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, 1)

today = date.today()

# ---------- Schema helpers ----------
@st.cache_data(show_spinner=False, ttl=120)
def get_current_db_name() -> str:
    df = run_query("SELECT DATABASE() AS db;")
    return df["db"].iat[0]

@st.cache_data(show_spinner=False, ttl=120)
def table_exists(table_name: str) -> bool:
    db = get_current_db_name()
    sql = """
    SELECT COUNT(*) AS cnt
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl
    """
    df = run_query(sql, {"db": db, "tbl": table_name})
    return df["cnt"].iat[0] > 0

@st.cache_data(show_spinner=False, ttl=120)
def column_exists(table_name: str, column_name: str) -> bool:
    db = get_current_db_name()
    sql = """
    SELECT COUNT(*) AS cnt
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl AND COLUMN_NAME = :col
    """
    df = run_query(sql, {"db": db, "tbl": table_name, "col": column_name})
    return df["cnt"].iat[0] > 0

# ======================================================
# Existing SQL: Q1–Q7, Q11, Q14
# ======================================================

# Q1: Products below reorder level
SQL_Q1 = """
SELECT  
    p.product_name, 
    p.stock_quantity, 
    p.reorder_level, 
    s.supplier_name, 
    (p.reorder_level - p.stock_quantity) AS units_to_order 
FROM products p 
JOIN suppliers s ON p.supplier_id = s.supplier_id 
WHERE p.stock_quantity < p.reorder_level 
ORDER BY units_to_order DESC;
"""

# Q2: Customers who haven't ordered in the last N days
def SQL_Q2(cutoff: date) -> Tuple[str, Dict[str, Any]]:
    return """
SELECT  
    c.customer_id, 
    c.first_name, 
    c.last_name, 
    c.email, 
    c.loyalty_tier, 
    MAX(o.order_date) AS last_order_date, 
    DATEDIFF(CURDATE(), MAX(o.order_date)) AS days_since_order 
FROM customers c 
LEFT JOIN orders o ON c.customer_id = o.customer_id 
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.loyalty_tier
HAVING (MAX(o.order_date) < :cutoff) OR MAX(o.order_date) IS NULL
ORDER BY days_since_order DESC;
""", {"cutoff": cutoff}

# Q3: Complete order details since start date
def SQL_Q3(start_dt: date) -> Tuple[str, Dict[str, Any]]:
    return """
SELECT  
    o.order_id, 
    CONCAT(c.first_name, ' ', c.last_name) AS customer_name, 
    c.email, 
    o.order_date, 
    p.product_name, 
    oi.quantity, 
    oi.unit_price, 
    oi.subtotal, 
    o.order_status 
FROM orders o 
INNER JOIN customers c ON o.customer_id = c.customer_id 
INNER JOIN order_items oi ON o.order_id = oi.order_id 
INNER JOIN products p ON oi.product_id = p.product_id 
WHERE o.order_date >= :start_dt
ORDER BY o.order_date DESC, o.order_id;
""", {"start_dt": start_dt}

# Q4: Revenue by product category (exclude Cancelled)
SQL_Q4 = """
SELECT  
    cat.category_name, 
    COUNT(DISTINCT oi.order_id) AS total_orders, 
    SUM(oi.quantity) AS units_sold, 
    SUM(oi.subtotal) AS total_revenue, 
    AVG(oi.unit_price) AS avg_selling_price, 
    SUM(oi.subtotal - (p.cost_price * oi.quantity)) AS profit 
FROM categories cat 
INNER JOIN products p ON cat.category_id = p.category_id 
INNER JOIN order_items oi ON p.product_id = oi.product_id 
INNER JOIN orders o ON oi.order_id = o.order_id 
WHERE o.order_status != 'Cancelled' 
GROUP BY cat.category_id, cat.category_name 
ORDER BY total_revenue DESC;
"""

# Q5: Products with no sales
SQL_Q5 = """
SELECT  
    p.product_id, 
    p.product_name, 
    p.stock_quantity, 
    c.category_name, 
    COUNT(oi.order_item_id) AS times_ordered 
FROM products p 
LEFT JOIN order_items oi ON p.product_id = oi.product_id 
LEFT JOIN categories c ON p.category_id = c.category_id 
GROUP BY p.product_id, p.product_name, p.stock_quantity, c.category_name 
HAVING times_ordered = 0 
ORDER BY p.stock_quantity DESC;
"""

# Q6: Customers without orders in a given year
def SQL_Q6(year: int) -> Tuple[str, Dict[str, Any]]:
    start, end = default_year_range(year)
    return """
SELECT  
    c.customer_id, 
    c.first_name, 
    c.last_name, 
    c.email, 
    c.registration_date, 
    c.loyalty_tier, 
    COUNT(o.order_id) AS order_count_year
FROM customers c 
LEFT JOIN orders o 
    ON c.customer_id = o.customer_id
    AND o.order_date >= :start_dt
    AND o.order_date < :end_dt
GROUP BY c.customer_id, c.first_name, c.last_name, c.email, c.registration_date, c.loyalty_tier
HAVING order_count_year = 0
ORDER BY c.customer_id;
""", {"start_dt": start, "end_dt": end}

# Q7: Employee hierarchy
SQL_Q7 = """
SELECT  
    e1.employee_id, 
    CONCAT(e1.first_name, ' ', e1.last_name) AS employee_name, 
    e1.department, 
    CONCAT(e2.first_name, ' ', e2.last_name) AS manager_name, 
    e2.department AS manager_department 
FROM employees e1 
LEFT JOIN employees e2 ON e1.manager_id = e2.employee_id 
ORDER BY e2.employee_id, e1.employee_id;
"""

# Q11: Top customers (above average spending)
SQL_Q11 = """
WITH customer_totals AS (
    SELECT 
        c.customer_id,
        CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
        c.loyalty_tier,
        COUNT(o.order_id) AS total_orders,
        COALESCE(SUM(o.total_amount), 0) AS total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id
),
avg_customer_spend AS (
    SELECT AVG(total_spent) AS avg_total_spent
    FROM customer_totals
),
avg_order_value AS (
    SELECT AVG(total_amount) AS aov FROM orders
)
SELECT 
    ct.customer_id,
    ct.customer_name,
    ct.loyalty_tier,
    ct.total_orders,
    ct.total_spent,
    aov.aov AS avg_order_value,
    (ct.total_spent - (aov.aov * ct.total_orders)) AS vs_average
FROM customer_totals ct
CROSS JOIN avg_customer_spend acs
CROSS JOIN avg_order_value aov
WHERE ct.total_spent > acs.avg_total_spent
ORDER BY ct.total_spent DESC;
"""

# Q14: Running total & 3-day moving average (MySQL 8+)
SQL_Q14 = """
SELECT  
    dt.order_day,
    dt.orders_count,
    dt.daily_revenue,
    SUM(dt.daily_revenue) OVER (ORDER BY dt.order_day) AS running_total, 
    AVG(dt.daily_revenue) OVER (
        ORDER BY dt.order_day 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg_3day
FROM (
    SELECT DATE(order_date) AS order_day,
           COUNT(order_id) AS orders_count,
           SUM(total_amount) AS daily_revenue
    FROM orders
    WHERE order_status != 'Cancelled'
    GROUP BY DATE(order_date)
) dt
ORDER BY dt.order_day;
"""

def sql_explain(sql: str) -> str:
    return f"EXPLAIN {sql}"

# ======================================================
# Q18: Monthly Sales Dashboard (with date range + optional status filter)
# ======================================================
def SQL_Q18(start_dt: date, end_dt: date, statuses: Optional[list] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Builds Q18 with dynamic IN clause if statuses provided.
    We expand placeholders to keep it safe & compatible with pandas.read_sql+SQLAlchemy.
    """
    params: Dict[str, Any] = {"start_dt": start_dt, "end_dt": end_dt}
    base = [
        "SELECT",
        "  DATE_FORMAT(o.order_date, '%Y-%m') AS month,",
        "  COUNT(DISTINCT o.order_id) AS total_orders,",
        "  COUNT(DISTINCT o.customer_id) AS unique_customers,",
        "  SUM(o.total_amount) AS revenue,",
        "  AVG(o.total_amount) AS avg_order_value,",
        "  SUM(CASE WHEN o.order_status = 'Cancelled' THEN 1 ELSE 0 END) AS cancelled_orders,",
        "  ROUND(SUM(CASE WHEN o.order_status = 'Cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS cancellation_rate",
        "FROM orders o",
        "WHERE o.order_date >= :start_dt",
        "  AND o.order_date < :end_dt",
    ]
    if statuses:
        placeholders = []
        for i, s in enumerate(statuses):
            key = f"status_{i}"
            placeholders.append(f":{key}")
            params[key] = s
        base.append(f"  AND o.order_status IN ({', '.join(placeholders)})")
    base += [
        "GROUP BY DATE_FORMAT(o.order_date, '%Y-%m')",
        "ORDER BY month DESC;",
    ]
    sql = "\n".join(base)
    return sql, params

# ======================================================
# NEW: Challenge SQL Builders (auto-detect schema)
# ======================================================

# ---------- Challenge 1: Inventory Management ----------
def SQL_C1_1_reorder_across_warehouses() -> Tuple[Optional[str], Dict[str, Any], str]:
    """
    Identify products that need reordering across all warehouses.
    Prefers warehouse_inventory if exists; else falls back to single-site products.
    """
    note = ""
    if table_exists("warehouse_inventory"):
        sql = """
        WITH prod_totals AS (
            SELECT wi.product_id,
                   SUM(wi.quantity) AS total_qty
            FROM warehouse_inventory wi
            GROUP BY wi.product_id
        )
        SELECT 
            p.product_id,
            p.product_name,
            COALESCE(pt.total_qty, 0) AS total_qty_across_warehouses,
            p.reorder_level,
            GREATEST(p.reorder_level - COALESCE(pt.total_qty, 0), 0) AS units_to_order
        FROM products p
        LEFT JOIN prod_totals pt ON p.product_id = pt.product_id
        WHERE COALESCE(pt.total_qty, 0) < p.reorder_level
        ORDER BY units_to_order DESC, p.product_name;
        """
        note = "Using warehouse_inventory to aggregate stock across all warehouses."
        return sql, {}, note
    else:
        note = "warehouse_inventory table not found; falling back to products.stock_quantity."
        return SQL_Q1, {}, note  # reuse existing below-reorder query

def SQL_C1_2_cost_to_restock() -> Tuple[Optional[str], Dict[str, Any], str]:
    """
    Calculate cost to restock products below reorder level to meet their target.
    """
    note = ""
    if table_exists("warehouse_inventory"):
        sql = """
        WITH prod_totals AS (
            SELECT wi.product_id,
                   SUM(wi.quantity) AS total_qty
            FROM warehouse_inventory wi
            GROUP BY wi.product_id
        ),
        deficits AS (
            SELECT p.product_id,
                   p.product_name,
                   COALESCE(pt.total_qty, 0) AS total_qty,
                   p.reorder_level,
                   GREATEST(p.reorder_level - COALESCE(pt.total_qty, 0), 0) AS deficit_units,
                   p.cost_price
            FROM products p
            LEFT JOIN prod_totals pt ON p.product_id = pt.product_id
            WHERE COALESCE(pt.total_qty, 0) < p.reorder_level
        )
        SELECT 
            SUM(deficit_units * cost_price) AS total_restock_cost,
            COUNT(*) AS products_below_level
        FROM deficits;
        """
        note = "Using total stock across warehouses to compute deficit * cost_price."
        return sql, {}, note
    else:
        sql = """
        SELECT 
            SUM((p.reorder_level - p.stock_quantity) * p.cost_price) AS total_restock_cost,
            COUNT(*) AS products_below_level
        FROM products p
        WHERE p.stock_quantity < p.reorder_level;
        """
        note = "warehouse_inventory not available; using products.stock_quantity as single-site."
        return sql, {}, note

def SQL_C1_3_warehouse_transfer_recommendations() -> Tuple[Optional[str], Dict[str, Any], str]:
    """
    Recommend transfers from surplus warehouses to deficit warehouses by product.
    """
    if not table_exists("warehouse_inventory"):
        return None, {}, "warehouse_inventory table is required for transfer recommendations."

    sql = """
    WITH diffs AS (
        SELECT 
            wi.warehouse_id,
            wi.product_id,
            SUM(wi.quantity) AS qty,
            p.reorder_level,
            SUM(wi.quantity) - p.reorder_level AS delta
        FROM warehouse_inventory wi
        JOIN products p ON p.product_id = wi.product_id
        GROUP BY wi.warehouse_id, wi.product_id, p.reorder_level
    ),
    surplus AS (
        SELECT warehouse_id, product_id, delta AS surplus_qty
        FROM diffs
        WHERE delta > 0
    ),
    deficit AS (
        SELECT warehouse_id, product_id, delta AS deficit_qty
        FROM diffs
        WHERE delta < 0
    )
    SELECT 
        p.product_id,
        p.product_name,
        s.warehouse_id AS from_warehouse,
        d.warehouse_id AS to_warehouse,
        LEAST(s.surplus_qty, -d.deficit_qty) AS transfer_qty
    FROM surplus s
    JOIN deficit d ON s.product_id = d.product_id
    JOIN products p ON p.product_id = s.product_id
    WHERE LEAST(s.surplus_qty, -d.deficit_qty) > 0
    ORDER BY p.product_id, transfer_qty DESC;
    """
    note = "Transfers computed from warehouses with surplus to those with deficit per product."
    return sql, {}, note

# ---------- Challenge 2: Customer Analytics ----------
def SQL_C2_1_cohort_registration_retention() -> Tuple[str, Dict[str, Any]]:
    """
    Registration-month cohorts and monthly retention (% of cohort active by order month).
    MySQL 8+ recommended for best performance.
    """
    sql = """
    WITH cust AS (
        SELECT c.customer_id,
               DATE_FORMAT(c.registration_date, '%Y-%m-01') AS cohort_month
        FROM customers c
    ),
    activity AS (
        SELECT o.customer_id,
               DATE_FORMAT(o.order_date, '%Y-%m-01') AS order_month
        FROM orders o
        WHERE o.order_status != 'Cancelled'
        GROUP BY o.customer_id, DATE_FORMAT(o.order_date, '%Y-%m-01')
    ),
    cohort_activity AS (
        SELECT 
            cust.cohort_month,
            act.order_month,
            TIMESTAMPDIFF(MONTH, cust.cohort_month, act.order_month) AS cohort_index,
            act.customer_id
        FROM cust
        JOIN activity act ON act.customer_id = cust.customer_id
        WHERE TIMESTAMPDIFF(MONTH, cust.cohort_month, act.order_month) >= 0
    ),
    cohort_sizes AS (
        SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_size
        FROM cust
        GROUP BY cohort_month
    ),
    cohort_counts AS (
        SELECT cohort_month, cohort_index, COUNT(DISTINCT customer_id) AS active_customers
        FROM cohort_activity
        GROUP BY cohort_month, cohort_index
    )
    SELECT 
        cs.cohort_month,
        cc.cohort_index,
        cs.cohort_size,
        cc.active_customers,
        ROUND(cc.active_customers * 100.0 / cs.cohort_size, 2) AS retention_pct
    FROM cohort_sizes cs
    LEFT JOIN cohort_counts cc
        ON cs.cohort_month = cc.cohort_month
    ORDER BY cs.cohort_month, cc.cohort_index;
    """
    return sql, {}

def SQL_C2_2_churn_rate_monthly() -> Tuple[str, Dict[str, Any]]:
    """
    Monthly churn = customers active in prev month but not in current / prev active.
    """
    sql = """
    WITH months AS (
        SELECT DISTINCT DATE_FORMAT(o.order_date, '%Y-%m-01') AS m
        FROM orders o
    ),
    active AS (
        SELECT DATE_FORMAT(o.order_date, '%Y-%m-01') AS m,
               o.customer_id
        FROM orders o
        WHERE o.order_status != 'Cancelled'
        GROUP BY DATE_FORMAT(o.order_date, '%Y-%m-01'), o.customer_id
    ),
    pairs AS (
        SELECT m AS curr_m, DATE_FORMAT(DATE_SUB(m, INTERVAL 1 MONTH), '%Y-%m-01') AS prev_m
        FROM months
    ),
    prev_set AS (
        SELECT p.curr_m, p.prev_m, a.customer_id
        FROM pairs p
        JOIN active a ON a.m = p.prev_m
    ),
    curr_set AS (
        SELECT p.curr_m, a.customer_id
        FROM (SELECT DISTINCT m AS curr_m FROM months) p
        LEFT JOIN active a ON a.m = p.curr_m
    ),
    churners AS (
        SELECT ps.curr_m, COUNT(*) AS churners
        FROM prev_set ps
        LEFT JOIN curr_set cs
          ON cs.curr_m = ps.curr_m AND cs.customer_id = ps.customer_id
        WHERE cs.customer_id IS NULL
        GROUP BY ps.curr_m
    ),
    prev_counts AS (
        SELECT p.curr_m, COUNT(*) AS prev_active
        FROM pairs p
        JOIN active a ON a.m = p.prev_m
        GROUP BY p.curr_m
    )
    SELECT 
        pc.curr_m AS month,
        pc.prev_active,
        COALESCE(c.churners, 0) AS churners,
        CASE WHEN pc.prev_active > 0
             THEN ROUND(100.0 * COALESCE(c.churners, 0) / pc.prev_active, 2)
             ELSE NULL END AS churn_rate_pct
    FROM prev_counts pc
    LEFT JOIN churners c ON c.curr_m = pc.curr_m
    ORDER BY month;
    """
    return sql, {}

def SQL_C2_3_upgrade_candidates(trailing_days: int = 90, threshold_pct: float = 0.8) -> Tuple[Optional[str], Dict[str, Any], str]:
    """
    Identify customers likely to upgrade loyalty tiers.
    If loyalty_tiers table exists with (tier_name, min_spend), use it.
    Otherwise, heuristics by percentile (top spenders) at current tier.
    """
    if table_exists("loyalty_tiers") and column_exists("loyalty_tiers", "min_spend"):
        sql = """
        WITH spend_window AS (
            SELECT 
                o.customer_id,
                SUM(o.total_amount) AS spend_lookback
            FROM orders o
            WHERE o.order_status != 'Cancelled'
              AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
            GROUP BY o.customer_id
        ),
        current_tier AS (
            SELECT c.customer_id, c.loyalty_tier, lt.min_spend AS current_min
            FROM customers c
            LEFT JOIN loyalty_tiers lt ON lt.tier_name = c.loyalty_tier
        ),
        next_tier AS (
            SELECT lt1.tier_name AS current_tier, lt2.tier_name AS next_tier, lt2.min_spend AS next_min
            FROM loyalty_tiers lt1
            JOIN loyalty_tiers lt2 ON lt2.min_spend > lt1.min_spend
        )
        SELECT 
            c.customer_id,
            CONCAT(cus.first_name, ' ', cus.last_name) AS customer_name,
            ct.loyalty_tier AS current_tier,
            nt.next_tier,
            sw.spend_lookback,
            nt.next_min,
            ROUND(100.0 * sw.spend_lookback / nt.next_min, 2) AS pct_to_next
        FROM current_tier ct
        JOIN customers cus ON cus.customer_id = ct.customer_id
        JOIN spend_window sw ON sw.customer_id = ct.customer_id
        JOIN next_tier nt ON nt.current_tier = ct.loyalty_tier
        WHERE sw.spend_lookback >= :threshold * nt.next_min
        ORDER BY pct_to_next DESC, sw.spend_lookback DESC
        LIMIT 200;
        """
        return sql, {"days": trailing_days, "threshold": threshold_pct}, "Using loyalty_tiers thresholds and last-N-days spend."
    else:
        # Heuristic: Rank customers within tier by last-N-days spend, pick top 10% as likely to upgrade
        sql = """
        WITH spend_window AS (
            SELECT 
                o.customer_id,
                SUM(o.total_amount) AS spend_lookback
            FROM orders o
            WHERE o.order_status != 'Cancelled'
              AND o.order_date >= DATE_SUB(CURDATE(), INTERVAL :days DAY)
            GROUP BY o.customer_id
        ),
        ranked AS (
            SELECT 
                c.customer_id,
                CONCAT(c.first_name, ' ', c.last_name) AS customer_name,
                c.loyalty_tier,
                COALESCE(sw.spend_lookback, 0) AS spend_lookback,
                PERCENT_RANK() OVER (PARTITION BY c.loyalty_tier ORDER BY COALESCE(sw.spend_lookback,0)) AS pr
            FROM customers c
            LEFT JOIN spend_window sw ON sw.customer_id = c.customer_id
        )
        SELECT *
        FROM ranked
        WHERE pr >= 0.90  -- top 10% by spend in their current tier
        ORDER BY loyalty_tier, spend_lookback DESC
        LIMIT 200;
        """
        return sql, {"days": trailing_days}, "Fallback heuristic: top 10% spenders within tier (last N days)."

# ---------- Challenge 3: Revenue Optimization ----------
def SQL_C3_1_profitable_combos(min_orders: int = 5) -> Tuple[str, Dict[str, Any]]:
    """
    Most profitable product pairs purchased together.
    """
    sql = """
    WITH item_profit AS (
        SELECT 
            oi.order_id,
            oi.product_id,
            SUM(oi.quantity) AS qty,
            SUM((oi.unit_price - p.cost_price) * oi.quantity) AS profit
        FROM order_items oi
        JOIN products p ON p.product_id = oi.product_id
        GROUP BY oi.order_id, oi.product_id
    )
    SELECT 
        ip1.product_id AS prod_a,
        ip2.product_id AS prod_b,
        COUNT(*) AS orders_together,
        SUM(ip1.profit + ip2.profit) AS total_pair_profit
    FROM item_profit ip1
    JOIN item_profit ip2 
      ON ip1.order_id = ip2.order_id AND ip1.product_id < ip2.product_id
    GROUP BY ip1.product_id, ip2.product_id
    HAVING COUNT(*) >= :min_orders
    ORDER BY total_pair_profit DESC, orders_together DESC
    LIMIT 100;
    """
    return sql, {"min_orders": int(min_orders)}

def SQL_C3_2_discount_effectiveness() -> Tuple[Optional[str], Dict[str, Any], str]:
    """
    Analyze discount effectiveness on revenue.
    Tries (in order):
      - order_items.discount_pct or discount_amount
      - products.list_price to infer % discount from list_price vs unit_price
      - else fallback notice
    """
    if column_exists("order_items", "discount_pct"):
        sql = """
        WITH enriched AS (
            SELECT 
                CASE 
                    WHEN discount_pct IS NULL THEN 0
                    WHEN discount_pct < 0 THEN 0
                    ELSE discount_pct
                END AS disc_pct,
                subtotal
            FROM order_items
        ),
        buckets AS (
            SELECT 
                CASE 
                    WHEN disc_pct = 0 THEN '0%'
                    WHEN disc_pct <= 10 THEN '0-10%'
                    WHEN disc_pct <= 20 THEN '10-20%'
                    WHEN disc_pct <= 30 THEN '20-30%'
                    ELSE '30%+'
                END AS discount_bucket,
                subtotal
            FROM enriched
        )
        SELECT discount_bucket,
               COUNT(*) AS line_items,
               SUM(subtotal) AS revenue,
               ROUND(AVG(subtotal), 2) AS avg_line_value
        FROM buckets
        GROUP BY discount_bucket
        ORDER BY 
            CASE discount_bucket 
                WHEN '0%' THEN 0
                WHEN '0-10%' THEN 1
                WHEN '10-20%' THEN 2
                WHEN '20-30%' THEN 3
                ELSE 4
            END;
        """
        return sql, {}, "Using order_items.discount_pct."
    elif column_exists("order_items", "discount_amount"):
        sql = """
        WITH enriched AS (
            SELECT 
                CASE 
                    WHEN unit_price <= 0 THEN 0
                    ELSE 100.0 * discount_amount / (unit_price + discount_amount)
                END AS disc_pct,
                (unit_price * quantity) AS revenue
            FROM order_items
        ),
        buckets AS (
            SELECT 
                CASE 
                    WHEN disc_pct = 0 THEN '0%'
                    WHEN disc_pct <= 10 THEN '0-10%'
                    WHEN disc_pct <= 20 THEN '10-20%'
                    WHEN disc_pct <= 30 THEN '20-30%'
                    ELSE '30%+'
                END AS discount_bucket,
                revenue
            FROM enriched
        )
        SELECT discount_bucket,
               COUNT(*) AS line_items,
               SUM(revenue) AS revenue,
               ROUND(AVG(revenue), 2) AS avg_line_value
        FROM buckets
        GROUP BY discount_bucket
        ORDER BY 
            CASE discount_bucket 
                WHEN '0%' THEN 0
                WHEN '0-10%' THEN 1
                WHEN '10-20%' THEN 2
                WHEN '20-30%' THEN 3
                ELSE 4
            END;
        """
        return sql, {}, "Using order_items.discount_amount derived %."
    elif column_exists("products", "list_price"):
        sql = """
        WITH joined AS (
            SELECT 
                oi.order_item_id,
                oi.unit_price,
                oi.quantity,
                p.list_price,
                CASE 
                    WHEN p.list_price > 0 THEN GREATEST(0, 100.0 * (p.list_price - oi.unit_price) / p.list_price)
                    ELSE 0
                END AS disc_pct,
                (oi.unit_price * oi.quantity) AS revenue
            FROM order_items oi
            JOIN products p ON p.product_id = oi.product_id
        ),
        buckets AS (
            SELECT 
                CASE 
                    WHEN disc_pct = 0 THEN '0%'
                    WHEN disc_pct <= 10 THEN '0-10%'
                    WHEN disc_pct <= 20 THEN '10-20%'
                    WHEN disc_pct <= 30 THEN '20-30%'
                    ELSE '30%+'
                END AS discount_bucket,
                revenue
            FROM joined
        )
        SELECT discount_bucket,
               COUNT(*) AS line_items,
               SUM(revenue) AS revenue,
               ROUND(AVG(revenue), 2) AS avg_line_value
        FROM buckets
        GROUP BY discount_bucket
        ORDER BY 
            CASE discount_bucket 
                WHEN '0%' THEN 0
                WHEN '0-10%' THEN 1
                WHEN '10-20%' THEN 2
                WHEN '20-30%' THEN 3
                ELSE 4
            END;
        """
        return sql, {}, "Inferring discount % from products.list_price vs order_items.unit_price."
    else:
        return None, {}, "No discount fields found (order_items.discount_* or products.list_price)."

def SQL_C3_3_revenue_per_warehouse() -> Tuple[Optional[str], Dict[str, Any], str]:
    """
    Calculate revenue per warehouse, trying common schema patterns:
      1) order_items.warehouse_id
      2) orders.warehouse_id
      3) shipments(order_item_id -> warehouse_id)
    """
    if column_exists("order_items", "warehouse_id"):
        sql = """
        SELECT 
            w.warehouse_id,
            w.warehouse_name,
            SUM(oi.subtotal) AS revenue
        FROM order_items oi
        JOIN orders o ON o.order_id = oi.order_id
        JOIN warehouses w ON w.warehouse_id = oi.warehouse_id
        WHERE o.order_status != 'Cancelled'
        GROUP BY w.warehouse_id, w.warehouse_name
        ORDER BY revenue DESC;
        """
        return sql, {}, "Using order_items.warehouse_id."
    elif column_exists("orders", "warehouse_id"):
        sql = """
        SELECT 
            w.warehouse_id,
            w.warehouse_name,
            SUM(o.total_amount) AS revenue
        FROM orders o
        JOIN warehouses w ON w.warehouse_id = o.warehouse_id
        WHERE o.order_status != 'Cancelled'
        GROUP BY w.warehouse_id, w.warehouse_name
        ORDER BY revenue DESC;
        """
        return sql, {}, "Using orders.warehouse_id."
    elif table_exists("shipments") and column_exists("shipments", "warehouse_id") and column_exists("shipments", "order_item_id"):
        sql = """
        SELECT 
            w.warehouse_id,
            w.warehouse_name,
            SUM(oi.subtotal) AS revenue
        FROM shipments s
        JOIN order_items oi ON oi.order_item_id = s.order_item_id
        JOIN orders o ON o.order_id = oi.order_id
        JOIN warehouses w ON w.warehouse_id = s.warehouse_id
        WHERE o.order_status != 'Cancelled'
        GROUP BY w.warehouse_id, w.warehouse_name
        ORDER BY revenue DESC;
        """
        return sql, {}, "Using shipments mapping of order_items to warehouses."
    else:
        return None, {}, "No warehouse mapping found (order_items/ orders/ shipments)."

# ---------- Challenge 4: Performance Utilities ----------
def SQL_C4_3_inactive_customers_join_version(cutoff: date) -> Tuple[str, Dict[str, Any]]:
    """
    Rewrites Q2 (HAVING with MAX) into anti-join for better performance.
    """
    sql = """
    SELECT 
        c.customer_id,
        c.first_name,
        c.last_name,
        c.email,
        c.loyalty_tier
    FROM customers c
    LEFT JOIN (
        SELECT DISTINCT o.customer_id
        FROM orders o
        WHERE o.order_date >= :cutoff
    ) recent ON recent.customer_id = c.customer_id
    WHERE recent.customer_id IS NULL
    ORDER BY c.customer_id;
    """
    return sql, {"cutoff": cutoff}

# ======================================================
# Sidebar Controls
# ======================================================
st.sidebar.header("Controls")
query_page = st.sidebar.radio(
    "View",
    ["📈 Dashboards", "📋 Tables / Queries", "🧩 Practice Challenges", "🧪 Performance (EXPLAIN)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

# Q3
param_start = st.sidebar.date_input("Start date for order details (Q3)", value=date(today.year, 1, 1))
# Q2
param_days = st.sidebar.number_input("Inactive window (days) for Q2", min_value=1, value=30, step=1)
# Q6
param_year = st.sidebar.number_input("Year for Q6", min_value=2000, max_value=2100, value=today.year, step=1)

# Q18
st.sidebar.markdown("---")
st.sidebar.subheader("Q18 – Monthly Sales")
default_end = date(today.year, today.month, 1)  # first of current month (exclusive end)
default_start = add_months(default_end, -12)
q18_start = st.sidebar.date_input("Q18 start (inclusive)", value=default_start, key="q18_start")
q18_end = st.sidebar.date_input("Q18 end (exclusive)", value=default_end, key="q18_end")
if q18_start >= q18_end:
    st.sidebar.error("Q18: End date must be after start date.")
status_filter_enabled = st.sidebar.checkbox("Q18: Filter by order status", value=False)
status_options = ["Pending", "Processing", "Shipped", "Delivered", "Cancelled", "Returned"]
selected_statuses = st.sidebar.multiselect(
    "Q18: Statuses",
    options=status_options,
    default=["Pending", "Processing", "Shipped", "Delivered", "Cancelled"],
    disabled=(not status_filter_enabled),
)

# Practice Challenge parameters
st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Practice Challenge Parameters")
pc_min_combo_orders = st.sidebar.number_input("C3.1: Min orders for product pair", min_value=1, value=5, step=1)
pc_trailing_days = st.sidebar.number_input("C2.3: Upgrade lookback days", min_value=7, value=90, step=1)
pc_threshold_pct = st.sidebar.slider("C2.3: % of next tier to flag", min_value=0.5, max_value=1.0, value=0.8, step=0.05)

# ======================================================
# Main Pages
# ======================================================
if query_page == "📈 Dashboards":
    st.subheader("Key Metrics & Visuals")

    # 1) Q4 – Revenue by Category
    with st.expander("💰 Revenue by Product Category (Q4)", expanded=True):
        with st.spinner("Fetching..."):
            df_cat = run_query(SQL_Q4, {})
        st.dataframe(df_cat, use_container_width=True)
        if not df_cat.empty:
            chart = (
                alt.Chart(df_cat)
                .mark_bar()
                .encode(
                    x=alt.X("total_revenue:Q", title="Total Revenue"),
                    y=alt.Y("category_name:N", sort="-x", title="Category"),
                    color=alt.Color("units_sold:Q", title="Units Sold", scale=alt.Scale(scheme="blues")),
                    tooltip=[
                        "category_name:N",
                        alt.Tooltip("total_orders:Q", title="Orders"),
                        alt.Tooltip("units_sold:Q", title="Units"),
                        alt.Tooltip("total_revenue:Q", title="Revenue", format=",.2f"),
                        alt.Tooltip("profit:Q", title="Profit", format=",.2f"),
                    ],
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)
            download_button(df_cat, "⬇️ Download Q4 (Revenue by Category)", "q4_revenue_by_category.csv")

    # 2) Q14 – Daily revenue with 3-day moving avg (MySQL 8+)
    with st.expander("📆 Daily Revenue – Running Total & 3-Day MA (Q14)", expanded=True):
        with st.spinner("Fetching..."):
            try:
                df_rt = run_query(SQL_Q14, {})
                df_rt = df_rt.sort_values("order_day")
                st.dataframe(df_rt, use_container_width=True)

                base = alt.Chart(df_rt).encode(x=alt.X("order_day:T", title="Date"))

                line_rev = base.mark_line(color="#1f77b4").encode(
                    y=alt.Y("daily_revenue:Q", title="Daily Revenue"),
                    tooltip=["order_day:T", alt.Tooltip("daily_revenue:Q", format=",.2f"), "orders_count:Q"],
                )
                line_ma = base.mark_line(color="#2ca02c").encode(
                    y=alt.Y("moving_avg_3day:Q", title="3-Day MA"),
                    tooltip=["order_day:T", alt.Tooltip("moving_avg_3day:Q", format=",.2f")],
                )

                st.altair_chart((line_rev + line_ma).resolve_scale(y="independent").properties(height=400), use_container_width=True)
                download_button(df_rt, "⬇️ Download Q14 (Daily Revenue & Running Totals)", "q14_daily_revenue_running.csv")

            except Exception:
                st.warning("Window functions require MySQL 8.0+. Falling back to Python-calculated running totals.")
                df_daily = run_query("""
                    SELECT DATE(order_date) AS order_day,
                           COUNT(order_id) AS orders_count,
                           SUM(total_amount) AS daily_revenue
                    FROM orders
                    WHERE order_status != 'Cancelled'
                    GROUP BY DATE(order_date)
                    ORDER BY DATE(order_date);
                """, {})
                if not df_daily.empty:
                    df_daily["running_total"] = df_daily["daily_revenue"].cumsum()
                    df_daily["moving_avg_3day"] = df_daily["daily_revenue"].rolling(3, min_periods=1).mean()
                    st.dataframe(df_daily, use_container_width=True)

                    base = alt.Chart(df_daily).encode(x=alt.X("order_day:T", title="Date"))
                    line1 = base.mark_line(color="#1f77b4").encode(y=alt.Y("daily_revenue:Q", title="Daily Revenue"))
                    line2 = base.mark_line(color="#2ca02c").encode(y=alt.Y("moving_avg_3day:Q", title="3-Day MA"))
                    st.altair_chart((line1 + line2).properties(height=400), use_container_width=True)
                    download_button(df_daily, "⬇️ Download Q14 Fallback", "q14_fallback_daily_revenue.csv")

    # 3) Q11 – Top customers
    with st.expander("🏅 Top Customers – Above Average Spend (Q11)", expanded=False):
        with st.spinner("Fetching..."):
            df_top = run_query(SQL_Q11, {})
        st.dataframe(df_top, use_container_width=True)
        if not df_top.empty:
            chart = (
                alt.Chart(df_top.head(20))
                .mark_bar(color="#8c564b")
                .encode(
                    x=alt.X("total_spent:Q", title="Total Spent"),
                    y=alt.Y("customer_name:N", sort="-x", title="Customer"),
                    tooltip=[
                        "customer_name:N",
                        "loyalty_tier:N",
                        alt.Tooltip("total_orders:Q", title="Orders"),
                        alt.Tooltip("total_spent:Q", title="Spent", format=",.2f"),
                        alt.Tooltip("avg_order_value:Q", title="AOV", format=",.2f"),
                    ],
                )
                .properties(height=500)
            )
            st.altair_chart(chart, use_container_width=True)
            download_button(df_top, "⬇️ Download Q11 (Top Customers)", "q11_top_customers.csv")

    # 4) Q1 – Low-stock products
    with st.expander("📦 Products Below Reorder Level (Q1)", expanded=False):
        with st.spinner("Fetching..."):
            df_low = run_query(SQL_Q1, {})
        st.dataframe(df_low, use_container_width=True)
        if not df_low.empty:
            bar = (
                alt.Chart(df_low.head(30))
                .mark_bar(color="#d62728")
                .encode(
                    x=alt.X("units_to_order:Q", title="Units to Order"),
                    y=alt.Y("product_name:N", sort="-x", title="Product"),
                    tooltip=["product_name:N", "supplier_name:N", "reorder_level:Q", "stock_quantity:Q", "units_to_order:Q"],
                )
                .properties(height=600)
            )
            st.altair_chart(bar, use_container_width=True)
            download_button(df_low, "⬇️ Download Q1 (Low Stock)", "q1_low_stock.csv")

    # 5) Q18 – Monthly Sales Dashboard
    with st.expander("📅 Monthly Sales Dashboard (Q18)", expanded=True):
        sql18, params18 = SQL_Q18(q18_start, q18_end, selected_statuses if status_filter_enabled else None)
        with st.spinner("Fetching monthly metrics..."):
            df_monthly = run_query(sql18, params=params18)

        if df_monthly.empty:
            st.info("Q18: No data for the selected period/filters.")
        else:
            df_monthly["month"] = pd.to_datetime(df_monthly["month"], format="%Y-%m")
            df_monthly = df_monthly.sort_values("month")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Revenue", f"{df_monthly['revenue'].sum():,.2f}")
            k2.metric("Total Orders", int(df_monthly["total_orders"].sum()))
            k3.metric("Unique Customers (sum)", int(df_monthly["unique_customers"].sum()))
            k4.metric("Avg Order Value (period avg)", f"{df_monthly['avg_order_value'].mean():,.2f}")

            st.markdown("**Monthly Table**")
            st.dataframe(df_monthly.assign(month=df_monthly["month"].dt.strftime("%Y-%m")), use_container_width=True)
            download_button(df_monthly.assign(month=df_monthly["month"].dt.strftime("%Y-%m")), "⬇️ Download Q18 (Monthly Metrics)", "q18_monthly_metrics.csv")

            st.markdown("**Charts**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Revenue by Month")
                chart_rev = (
                    alt.Chart(df_monthly)
                    .mark_bar(color="#1f77b4")
                    .encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y("revenue:Q", title="Revenue"),
                        tooltip=[
                            alt.Tooltip("month:T", format="%Y-%m", title="Month"),
                            alt.Tooltip("revenue:Q", format=",.2f", title="Revenue"),
                            alt.Tooltip("total_orders:Q", title="Orders"),
                            alt.Tooltip("avg_order_value:Q", format=",.2f", title="AOV"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_rev, use_container_width=True)
            with c2:
                st.caption("Orders & Unique Customers")
                base = alt.Chart(df_monthly).encode(x=alt.X("month:T", title="Month"))
                line_o = base.mark_line(color="#ff7f0e", point=True).encode(
                    y=alt.Y("total_orders:Q", title="Orders"),
                    tooltip=[alt.Tooltip("month:T", format="%Y-%m"), "total_orders:Q"],
                )
                line_c = base.mark_line(color="#2ca02c", point=True).encode(
                    y=alt.Y("unique_customers:Q", title="Customers"),
                    tooltip=[alt.Tooltip("month:T", format="%Y-%m"), "unique_customers:Q"],
                )
                st.altair_chart((line_o + line_c).resolve_scale(y="independent").properties(height=320), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                st.caption("Average Order Value (AOV)")
                chart_aov = (
                    alt.Chart(df_monthly)
                    .mark_line(color="#9467bd", point=True)
                    .encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y("avg_order_value:Q", title="AOV"),
                        tooltip=[alt.Tooltip("month:T", format="%Y-%m"), alt.Tooltip("avg_order_value:Q", format=",.2f")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_aov, use_container_width=True)
            with c4:
                st.caption("Cancellation Rate (%)")
                chart_cr = (
                    alt.Chart(df_monthly)
                    .mark_line(color="#d62728", point=True)
                    .encode(
                        x=alt.X("month:T", title="Month"),
                        y=alt.Y("cancellation_rate:Q", title="Cancellation Rate (%)"),
                        tooltip=[alt.Tooltip("month:T", format="%Y-%m"), alt.Tooltip("cancellation_rate:Q", format=".2f")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart_cr, use_container_width=True)

            with st.expander("🔎 Show SQL (Q18)"):
                st.code(sql18, language="sql")
                st.json(params18)

elif query_page == "📋 Tables / Queries":
    st.subheader("Query Explorer")

    tabs = st.tabs([
        "Q1 Low Stock",
        "Q2 Inactive Customers",
        "Q3 Order Details",
        "Q4 Revenue by Category",
        "Q5 No-Sales Products",
        "Q6 No Orders in Year",
        "Q7 Employee Hierarchy",
        "Q11 Top Customers",
        "Q14 Daily Revenue (Window)",
        "Q18 Monthly Sales",
    ])

    with tabs[0]:
        st.markdown("### Q1: Products below reorder level")
        with st.spinner("Fetching..."):
            df = run_query(SQL_Q1, {})
        st.dataframe(df, use_container_width=True)
        download_button(df, "⬇️ Download Q1", "q1_low_stock.csv")

    with tabs[1]:
        st.markdown("### Q2: Customers inactive past N days")
        cutoff = today - timedelta(days=int(param_days))
        sql, params = SQL_Q2(cutoff)
        with st.spinner("Fetching..."):
            df = run_query(sql, params)
        st.caption(f"Cutoff date: {cutoff.isoformat()}")
        st.dataframe(df, use_container_width=True)
        download_button(df, "⬇️ Download Q2", "q2_inactive_customers.csv")

    with tabs[2]:
        st.markdown("### Q3: Complete order details since a date")
        sql, params = SQL_Q3(param_start)
        with st.spinner("Fetching..."):
            df = run_query(sql, params)
        st.caption(f"Start date: {param_start.isoformat()}")
        st.dataframe(df, use_container_width=True)
        download_button(df, "⬇️ Download Q3", "q3_order_details.csv")

    with tabs[3]:
        st.markdown("### Q4: Revenue by product category")
        with st.spinner("Fetching..."):
            df = run_query(SQL_Q4, {})
        st.dataframe(df, use_container_width=True)
        download_button(df, "⬇️ Download Q4", "q4_revenue_by_category.csv")

    with tabs[4]:
        st.markdown("### Q5: Products with no sales")
        with st.spinner("Fetching..."):
            df = run_query(SQL_Q5, {})
        st.dataframe(df, use_container_width=True)
        download_button(df, "⬇️ Download Q5", "q5_products_no_sales.csv")

    with tabs[5]:
        st.markdown("### Q6: Customers without orders in selected year")
        sql, params = SQL_Q6(int(param_year))
        with st.spinner("Fetching..."):
            df = run_query(sql, params)
        st.caption(f"Year: {param_year}")
        st.dataframe(df, use_container_width=True)
        download_button(df, "⬇️ Download Q6", "q6_customers_no_orders_year.csv")

    with tabs[6]:
        st.markdown("### Q7: Employee hierarchy")
        with st.spinner("Fetching..."):
            df = run_query(SQL_Q7, {})
        st.dataframe(df, use_container_width=True)
        download_button(df, "⬇️ Download Q7", "q7_employee_hierarchy.csv")

    with tabs[7]:
        st.markdown("### Q11: Top customers above average spending")
        with st.spinner("Fetching..."):
            df = run_query(SQL_Q11, {})
        st.dataframe(df, use_container_width=True)
        download_button(df, "⬇️ Download Q11", "q11_top_customers.csv")

    with tabs[8]:
        st.markdown("### Q14: Running total of daily revenue (MySQL 8+)")
        try:
            with st.spinner("Fetching..."):
                df = run_query(SQL_Q14, {})
            st.dataframe(df, use_container_width=True)
            download_button(df, "⬇️ Download Q14", "q14_running_daily_revenue.csv")
        except Exception:
            st.error("This query requires MySQL 8.0+ (window functions). See Dashboards for Python fallback.")

    with tabs[9]:
        st.markdown("### Q18: Monthly Sales Dashboard")
        sql18, params18 = SQL_Q18(q18_start, q18_end, selected_statuses if status_filter_enabled else None)
        with st.spinner("Fetching..."):
            df = run_query(sql18, params18)
        if df.empty:
            st.info("No data for the selected Q18 period/filters.")
        else:
            st.dataframe(df, use_container_width=True)
            download_button(df, "⬇️ Download Q18", "q18_monthly_metrics.csv")

elif query_page == "🧩 Practice Challenges":
    st.subheader("Phase 7 – Practice Challenges")

    # ---------------- Challenge 1 ----------------
    with st.expander("Challenge 1: Inventory Management", expanded=True):
        st.markdown("**1) Identify products that need reordering across all warehouses**")
        sql, params, note = SQL_C1_1_reorder_across_warehouses()
        st.caption(note)
        with st.spinner("Running..."):
            df = run_query(sql, params) if sql else pd.DataFrame()
        if df.empty:
            st.info("No products below reorder level (or no data).")
        else:
            st.dataframe(df, use_container_width=True)
            download_button(df, "⬇️ Download C1.1", "c1_1_reorder_across_warehouses.csv")
        with st.expander("Show SQL (C1.1)"):
            st.code(sql or "-- requires warehouse_inventory", language="sql")

        st.markdown("---")
        st.markdown("**2) Calculate the cost of restocking all products below reorder level**")
        sql2, p2, note2 = SQL_C1_2_cost_to_restock()
        st.caption(note2)
        with st.spinner("Running..."):
            df2 = run_query(sql2, p2) if sql2 else pd.DataFrame()
        st.dataframe(df2, use_container_width=True)
        download_button(df2, "⬇️ Download C1.2", "c1_2_cost_to_restock.csv")
        with st.expander("Show SQL (C1.2)"):
            st.code(sql2 or "-- not available", language="sql")

        st.markdown("---")
        st.markdown("**3) Recommend warehouse transfers to balance inventory**")
        sql3, p3, note3 = SQL_C1_3_warehouse_transfer_recommendations()
        st.caption(note3)
        if sql3:
            with st.spinner("Running..."):
                df3 = run_query(sql3, p3)
            if df3.empty:
                st.info("No transfer recommendations (balanced or insufficient data).")
            else:
                st.dataframe(df3, use_container_width=True)
                download_button(df3, "⬇️ Download C1.3", "c1_3_warehouse_transfers.csv")
            with st.expander("Show SQL (C1.3)"):
                st.code(sql3, language="sql")
        else:
            st.info("warehouse_inventory is required for transfers.")

    # ---------------- Challenge 2 ----------------
    with st.expander("Challenge 2: Customer Analytics", expanded=True):
        st.markdown("**1) Customer cohort analysis by registration month (retention)**")
        sqlc, pc = SQL_C2_1_cohort_registration_retention()
        with st.spinner("Running..."):
            dfc = run_query(sqlc, pc)
        st.dataframe(dfc, use_container_width=True)
        if not dfc.empty:
            heat = (
                alt.Chart(dfc)
                .mark_rect()
                .encode(
                    x=alt.X("cohort_index:O", title="Cohort Month Index"),
                    y=alt.Y("cohort_month:O", title="Cohort (YYYY-MM)"),
                    color=alt.Color("retention_pct:Q", title="Retention %", scale=alt.Scale(scheme="greens")),
                    tooltip=[
                        alt.Tooltip("cohort_month:N", title="Cohort"),
                        alt.Tooltip("cohort_index:Q", title="Index"),
                        alt.Tooltip("cohort_size:Q", title="Size"),
                        alt.Tooltip("active_customers:Q", title="Active"),
                        alt.Tooltip("retention_pct:Q", title="Retention %"),
                    ],
                )
            )
            st.altair_chart(heat, use_container_width=True)
            download_button(dfc, "⬇️ Download C2.1", "c2_1_cohort_retention.csv")
        with st.expander("Show SQL (C2.1)"):
            st.code(sqlc, language="sql")

        st.markdown("---")
        st.markdown("**2) Calculate customer churn rate (monthly)**")
        sqlch, pch = SQL_C2_2_churn_rate_monthly()
        with st.spinner("Running..."):
            dfch = run_query(sqlch, pch)
        st.dataframe(dfch, use_container_width=True)
        if not dfch.empty:
            line = (
                alt.Chart(dfch)
                .mark_line(point=True, color="#d62728")
                .encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("churn_rate_pct:Q", title="Churn %")
                )
                .properties(height=300)
            )
            st.altair_chart(line, use_container_width=True)
            download_button(dfch, "⬇️ Download C2.2", "c2_2_churn_rate.csv")
        with st.expander("Show SQL (C2.2)"):
            st.code(sqlch, language="sql")

        st.markdown("---")
        st.markdown("**3) Identify customers likely to upgrade loyalty tiers**")
        sqlu, pu, noteu = SQL_C2_3_upgrade_candidates(trailing_days=int(pc_trailing_days), threshold_pct=float(pc_threshold_pct))
        st.caption(noteu)
        if sqlu:
            with st.spinner("Running..."):
                dfu = run_query(sqlu, pu)
            st.dataframe(dfu, use_container_width=True)
            download_button(dfu, "⬇️ Download C2.3", "c2_3_upgrade_candidates.csv")
            with st.expander("Show SQL (C2.3)"):
                st.code(sqlu, language="sql")
        else:
            st.info("Unable to build upgrade query (check schema).")

    # ---------------- Challenge 3 ----------------
    with st.expander("Challenge 3: Revenue Optimization", expanded=True):
        st.markdown("**1) Find the most profitable product combinations**")
        sqlp, pp = SQL_C3_1_profitable_combos(int(pc_min_combo_orders))
        with st.spinner("Running..."):
            dfp = run_query(sqlp, pp)
        st.dataframe(dfp, use_container_width=True)
        download_button(dfp, "⬇️ Download C3.1", "c3_1_profitable_pairs.csv")
        with st.expander("Show SQL (C3.1)"):
            st.code(sqlp, language="sql")

        st.markdown("---")
        st.markdown("**2) Analyze discount effectiveness on revenue**")
        sqld, pd_, noted = SQL_C3_2_discount_effectiveness()
        st.caption(noted)
        if sqld:
            with st.spinner("Running..."):
                dfd = run_query(sqld, pd_)
            st.dataframe(dfd, use_container_width=True)
            if not dfd.empty:
                bar = (
                    alt.Chart(dfd)
                    .mark_bar(color="#1f77b4")
                    .encode(
                        x=alt.X("discount_bucket:N", title="Discount Bucket"),
                        y=alt.Y("revenue:Q", title="Revenue"),
                        tooltip=["discount_bucket:N", alt.Tooltip("revenue:Q", format=",.2f"), "line_items:Q", alt.Tooltip("avg_line_value:Q", format=",.2f")]
                    )
                )
                st.altair_chart(bar, use_container_width=True)
            download_button(dfd, "⬇️ Download C3.2", "c3_2_discount_effectiveness.csv")
            with st.expander("Show SQL (C3.2)"):
                st.code(sqld, language="sql")
        else:
            st.info("No discount fields found to analyze.")

        st.markdown("---")
        st.markdown("**3) Calculate revenue per warehouse**")
        sqlw, pw, notew = SQL_C3_3_revenue_per_warehouse()
        st.caption(notew)
        if sqlw:
            with st.spinner("Running..."):
                dfw = run_query(sqlw, pw)
            st.dataframe(dfw, use_container_width=True)
            download_button(dfw, "⬇️ Download C3.3", "c3_3_revenue_per_warehouse.csv")
            with st.expander("Show SQL (C3.3)"):
                st.code(sqlw, language="sql")
        else:
            st.info("Warehouse mapping columns/tables not found to compute revenue per warehouse.")

    # ---------------- Challenge 4 ----------------
    with st.expander("Challenge 4: Performance Tuning", expanded=True):
        st.markdown("**1) Optimize the slowest queries using EXPLAIN**")
        target = st.selectbox(
            "Pick a Challenge query to EXPLAIN",
            ["C1.1", "C1.2", "C1.3", "C2.1", "C2.2", "C2.3", "C3.1", "C3.2", "C3.3"]
        )

        explain_sql = None
        explain_params: Dict[str, Any] = {}
        if target == "C1.1":
            s, p, _ = SQL_C1_1_reorder_across_warehouses()
            explain_sql, explain_params = s, p
        elif target == "C1.2":
            s, p, _ = SQL_C1_2_cost_to_restock()
            explain_sql, explain_params = s, p
        elif target == "C1.3":
            s, p, _ = SQL_C1_3_warehouse_transfer_recommendations()
            explain_sql, explain_params = s, p
        elif target == "C2.1":
            s, p = SQL_C2_1_cohort_registration_retention()
            explain_sql, explain_params = s, p
        elif target == "C2.2":
            s, p = SQL_C2_2_churn_rate_monthly()
            explain_sql, explain_params = s, p
        elif target == "C2.3":
            s, p, _ = SQL_C2_3_upgrade_candidates(trailing_days=int(pc_trailing_days), threshold_pct=float(pc_threshold_pct))
            explain_sql, explain_params = s, p
        elif target == "C3.1":
            s, p = SQL_C3_1_profitable_combos(int(pc_min_combo_orders))
            explain_sql, explain_params = s, p
        elif target == "C3.2":
            s, p, _ = SQL_C3_2_discount_effectiveness()
            explain_sql, explain_params = s, p
        elif target == "C3.3":
            s, p, _ = SQL_C3_3_revenue_per_warehouse()
            explain_sql, explain_params = s, p

        if explain_sql:
            st.code(explain_sql, language="sql")
            if st.button("Run EXPLAIN (Challenge)", type="primary"):
                with st.spinner("Explaining challenge query plan..."):
                    df = run_query(sql_explain(explain_sql), explain_params)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("Selected query not available due to schema constraints.")

        st.markdown("---")
        st.markdown("**2) Create appropriate indexes**")
        st.write("""
- Suggested composite indexes that match joins and filters:
  - `orders(customer_id, order_date)` — Q3, Q6, churn/cohorts.
  - `order_items(order_id, product_id)` and `order_items(product_id, order_id)` — pairs, category revenue.
  - `products(category_id)` — Q4.
  - `warehouse_inventory(product_id, warehouse_id)` — C1 transfers/aggregation.
  - `orders(order_date)` — Q14/Q18.
  - If using status filters often: `orders(order_status, order_date)` (watch selectivity).
- Avoid `FUNCTION(order_date)` in WHERE; always use ranges.
        """)

        idx_sql_examples = [
            "CREATE INDEX idx_orders_customer_date ON orders(customer_id, order_date);",
            "CREATE INDEX idx_order_items_order_product ON order_items(order_id, product_id);",
            "CREATE INDEX idx_order_items_product_order ON order_items(product_id, order_id);",
            "CREATE INDEX idx_products_category ON products(category_id);",
            "CREATE INDEX idx_orders_status_date ON orders(order_status, order_date);",
            "CREATE INDEX idx_warehouse_inventory_prod_wh ON warehouse_inventory(product_id, warehouse_id);"
        ]
        with st.expander("Show DDL"):
            st.code("\n".join(idx_sql_examples), language="sql")

        apply_ok = st.checkbox("I understand DDL will modify my database. Proceed to apply selected indexes.")
        chosen = st.multiselect("Pick indexes to apply", options=idx_sql_examples, default=[])
        if st.button("Apply Indexes") and apply_ok and chosen:
            with st.spinner("Applying indexes..."):
                for stmt in chosen:
                    try:
                        exec_sql(stmt)
                        st.success(f"Applied: {stmt}")
                    except Exception as e:
                        st.error(f"Failed: {stmt}\n{e}")

        st.markdown("---")
        st.markdown("**3) Rewrite a subquery using JOINs for better performance**")
        cutoff = today - timedelta(days=int(param_days))
        st.caption(f"Cutoff date: {cutoff.isoformat()}")
        q2_original, params_orig = SQL_Q2(cutoff)
        q2_joined, params_joined = SQL_C4_3_inactive_customers_join_version(cutoff)
        choice = st.radio("Version", ["Original (GROUP BY + HAVING)", "Rewritten (Anti-Join)"], horizontal=True)
        if choice.startswith("Original"):
            st.code(q2_original, language="sql")
            with st.spinner("Running original..."):
                dfv = run_query(q2_original, params_orig)
        else:
            st.code(q2_joined, language="sql")
            with st.spinner("Running rewritten..."):
                dfv = run_query(q2_joined, params_joined)
        st.dataframe(dfv, use_container_width=True)
        download_button(dfv, "⬇️ Download C4.3", "c4_3_inactive_customers_version.csv")

else:
    st.subheader("🧪 Query Performance & EXPLAIN")
    explain_target = st.selectbox(
        "Select a query to EXPLAIN",
        ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q11", "Q14", "Q18"]
    )

    sql_map_plain = {
        "Q1": (SQL_Q1, {}),
        "Q4": (SQL_Q4, {}),
        "Q5": (SQL_Q5, {}),
        "Q7": (SQL_Q7, {}),
        "Q11": (SQL_Q11, {}),
        "Q14": (SQL_Q14, {}),
    }

    # Build SQL + params for parameterized queries
    if explain_target == "Q2":
        cutoff = today - timedelta(days=int(param_days))
        sql, params = SQL_Q2(cutoff)
    elif explain_target == "Q3":
        sql, params = SQL_Q3(param_start)
    elif explain_target == "Q6":
        sql, params = SQL_Q6(int(param_year))
    elif explain_target == "Q18":
        sql, params = SQL_Q18(q18_start, q18_end, selected_statuses if status_filter_enabled else None)
    else:
        sql, params = sql_map_plain[explain_target]

    st.code(sql, language="sql")

    if st.button("Run EXPLAIN", type="primary"):
        with st.spinner("Explaining query plan..."):
            df = run_query(sql_explain(sql), params)
        st.dataframe(df, use_container_width=True)

    st.markdown("### Index Suggestions & Tips")
    st.markdown("""
- **Composite indexes** often help join + filter patterns:
  - `orders(customer_id, order_date)` for Q3, Q6.
  - `order_items(product_id, order_id)` for Q4, Q5.
  - `products(category_id)` for Q4.
- Avoid `YEAR(order_date) = ?` in WHERE; prefer range: `order_date >= 'YYYY-01-01' AND < 'YYYY+1-01-01'` (Q6 does this in JOIN).
- For low-selectivity columns (e.g., status), indexes may be less effective alone.
- Use `EXPLAIN` to verify index usage; ensure `type` is not `ALL` on big tables.
""")