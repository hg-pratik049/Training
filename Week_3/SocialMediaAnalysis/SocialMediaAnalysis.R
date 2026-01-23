

library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(DT)
library(scales)
library(lubridate)
library(zoo)


social_data <- read_csv("dataset.csv", show_col_types = FALSE)

social_data <- social_data %>%
  mutate(
    date  = as.Date(parse_date_time(timestamp,
                                    orders = c("Y-m-d H:M:S z", "Y-m-d H:M:S"))),
    retweet_count = ifelse(!is.na(retweeted_status_id), 1L, 0L),
    likes = rating_numerator
  )

I
daily_data_base <- social_data %>%
  group_by(date) %>%
  summarise(
    likes    = sum(likes, na.rm = TRUE),
    retweets = sum(retweet_count, na.rm = TRUE),
    tweets   = n(),
    .groups  = "drop"
  ) %>%
  arrange(date) %>%
  mutate(
    likes_ma7    = zoo::rollmean(likes,    7, fill = NA, align = "right"),
    retweets_ma7 = zoo::rollmean(retweets, 7, fill = NA, align = "right"),
    tweets_ma7   = zoo::rollmean(tweets,   7, fill = NA, align = "right"),
    
    
    spend = round(runif(n(), 50, 150), 2),
    revenue = round(tweets * runif(n(), 3, 10), 2)
  )


ui <- dashboardPage(
  dashboardHeader(title = "Social Media Analytics"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("dashboard")),
      menuItem("ROI Analysis", tabName = "roi", icon = icon("chart-line")),
      menuItem("Details", tabName = "details", icon = icon("table"))
    ),
    dateRangeInput(
      "date_range", "Date range",
      start = min(daily_data_base$date), end = max(daily_data_base$date)
    )
  ),
  
  dashboardBody(
    fluidRow(
      valueBoxOutput("vb_total_likes", width = 4),
      valueBoxOutput("vb_total_retweets", width = 4),
      valueBoxOutput("vb_total_tweets", width = 4)
    ),
    
    tabItems(
      
      tabItem(tabName = "overview",
              fluidRow(
                box(
                  title = "Likes Over Time (7‑day MA)", status = "primary",
                  solidHeader = TRUE, collapsible = TRUE,
                  plotOutput("likesPlot", height = 280), width = 12
                )
              ),
              fluidRow(
                box(
                  title = "Retweets Over Time (7‑day MA)", status = "primary",
                  solidHeader = TRUE, collapsible = TRUE,
                  plotOutput("retweetsPlot", height = 280), width = 6
                ),
                box(
                  title = "Tweets Over Time (7‑day MA)", status = "primary",
                  solidHeader = TRUE, collapsible = TRUE,
                  plotOutput("tweetsPlot", height = 280), width = 6
                )
              ),
              fluidRow(
                box(
                  title = "Daily Composition: Likes vs Retweets", status = "primary",
                  solidHeader = TRUE, collapsible = TRUE,
                  plotOutput("stackedArea", height = 280), width = 12
                )
              )
      ),
      
      
      tabItem(tabName = "roi",
              fluidRow(
                valueBoxOutput("vb_spend", width = 6),
                valueBoxOutput("vb_revenue", width = 6)
              ),
              fluidRow(
                box(
                  title = "Daily Spend vs Revenue (Synthetic Demo)",
                  status = "primary", solidHeader = TRUE, width = 12,
                  plotOutput("roiPlot", height = 300)
                )
              )
      ),
      
      
      tabItem(tabName = "details",
              fluidRow(
                box(
                  title = "Detailed Metrics", status = "primary",
                  solidHeader = TRUE, collapsible = TRUE, width = 12,
                  dataTableOutput("detailsTable")
                )
              )
      )
    )
  )
)


server <- function(input, output, session) {
  
  daily_data <- reactive({
    daily_data_base %>%
      filter(date >= input$date_range[1], date <= input$date_range[2])
  })
  
  social_filtered <- reactive({
    social_data %>%
      filter(date >= input$date_range[1], date <= input$date_range[2])
  })
  
  
  output$vb_total_likes <- renderValueBox({
    dd <- daily_data()
    valueBox(
      label_comma()(sum(dd$likes)), "Total Likes (proxy)",
      icon = icon("thumbs-up"), color = "blue"
    )
  })
  
  output$vb_total_retweets <- renderValueBox({
    dd <- daily_data()
    valueBox(
      label_comma()(sum(dd$retweets)), "Total Retweets (proxy)",
      icon = icon("retweet"), color = "green"
    )
  })
  
  output$vb_total_tweets <- renderValueBox({
    dd <- daily_data()
    valueBox(
      label_comma()(sum(dd$tweets)), "Total Tweets",
      icon = icon("hashtag"), color = "purple"
    )
  })
  
  
  output$vb_spend <- renderValueBox({
    dd <- daily_data()
    valueBox(
      paste0("$", label_comma()(sum(dd$spend))),
      "Total Spend (Synthetic)",
      icon = icon("credit-card"), color = "yellow"
    )
  })
  
  output$vb_revenue <- renderValueBox({
    dd <- daily_data()
    valueBox(
      paste0("$", label_comma()(sum(dd$revenue))),
      "Total Revenue (Synthetic)",
      icon = icon("dollar-sign"), color = "olive"
    )
  })
  
  
  output$roiPlot <- renderPlot({
    dd <- daily_data()
    ggplot(dd, aes(x = date)) +
      geom_col(aes(y = spend), fill = "#F9A825", alpha = 0.7) +
      geom_line(aes(y = revenue), color = "#0D47A1", linewidth = 1.2) +
      labs(x = NULL, y = "USD",
           subtitle = "Bars = Spend, Line = Revenue (Synthetic)") +
      theme_minimal()
  })
  
  
  output$likesPlot <- renderPlot({
    dd <- daily_data()
    ggplot(dd, aes(date, likes)) +
      geom_col(fill = "#90CAF9") +
      geom_line(aes(y = likes_ma7), color = "#0D47A1", linewidth = 1) +
      theme_minimal()
  })
  
  output$retweetsPlot <- renderPlot({
    dd <- daily_data()
    ggplot(dd, aes(date, retweets)) +
      geom_col(fill = "#A5D6A7") +
      geom_line(aes(y = retweets_ma7), color = "#1B5E20", linewidth = 1) +
      theme_minimal()
  })
  
  output$tweetsPlot <- renderPlot({
    dd <- daily_data()
    ggplot(dd, aes(date, tweets)) +
      geom_col(fill = "#CE93D8") +
      geom_line(aes(y = tweets_ma7), color = "#4A148C", linewidth = 1) +
      theme_minimal()
  })
  
  output$stackedArea <- renderPlot({
    dd <- daily_data() %>%
      pivot_longer(cols = c(likes, retweets), names_to = "metric", values_to = "value")
    ggplot(dd, aes(date, value, fill = metric)) +
      geom_area(alpha = 0.8) +
      scale_fill_manual(values = c("likes" = "#64B5F6", "retweets" = "#81C784")) +
      theme_minimal()
  })
  
  
  output$detailsTable <- renderDataTable({
    datatable(
      social_filtered() %>%
        transmute(
          date, tweet_id, text, likes,
          is_retweet = ifelse(retweet_count == 1, "Yes", "No"),
          doggo, floofer, pupper, puppo, name
        ),
      filter = "top", options = list(pageLength = 10, scrollX = TRUE)
    )
  })
  
}


shinyApp(ui = ui, server = server)
