import pandas as pd
import re
from mcp.server.fastmcp import FastMCP

from typing import Optional

# Load the dataset
df = pd.read_csv("/Users/sithijshetty/Desktop/Online-retail/online_retail.csv")
# Convert InvoiceDate to datetime for proper filtering
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

# Create an MCP server
mcp = FastMCP(
    name="Sales Analysis",
    host="0.0.0.0",
    port=8050,
)

SYSTEM_PROMPT = """
You are an intelligent assistant analyzing an online retail sales dataset with the following key fields:

- InvoiceDate: Date and time of each transaction.
- Country: Country where purchase was made.
- Description: Product description (contains product category keywords).
- Quantity and UnitPrice: Used to calculate sales/revenue.

Your task:
- Parse the user's query to identify which tool best fits their request.
- Extract relevant parameters such as:
  - Country (e.g., "United Kingdom", "France")
  - Category keywords from the Description (e.g., "mug", "lantern")
  - Granularity (week, month, quarter)
  - N for top products
  - Date ranges (e.g., last 3 months, between dates)
- Respond ONLY by selecting the most appropriate tool and invoking it with correct parameters.
- Choose the chart type that best illustrates the data requested.
- If no sufficient parameters are provided, ask the user for clarification.

Example requests:
- "Show me the monthly sales trend of mugs in France."
- "List the top 10 products sold in the last quarter."
- "Give me the product category sales distribution."

Follow these rules strictly to ensure accurate tool usage and meaningful visualization.
After receiving tool outputs, respond with a concise, final answer to the user without repeating the tool call process.
"""

@mcp.prompt()
def get_system_prompt() -> str:
    return SYSTEM_PROMPT

# Tool 1: Dynamic Sales Trend Over Time
@mcp.tool()
def dynamic_sales_trend(
    country: Optional[str] = None,
    category: Optional[str] = None,
    granularity: Optional[str] = "month"
) -> dict:
    """
    Generate a dynamic sales trend over time with optional filters.
    Returns both a text summary and raw data for charting.
    """
    filtered_df = df.copy()

    if country:
        filtered_df = filtered_df[filtered_df['Country'] == country]

    if category:
        filtered_df = filtered_df[
            filtered_df['Description'].str.contains(category, case=False, na=False)
        ]

    if granularity not in ["week", "month", "quarter"]:
        granularity = "month"

    if granularity == "week":
        filtered_df['Period'] = filtered_df['InvoiceDate'].dt.to_period('W')
    elif granularity == "quarter":
        filtered_df['Period'] = filtered_df['InvoiceDate'].dt.to_period('Q')
    else:
        filtered_df['Period'] = filtered_df['InvoiceDate'].dt.to_period('M')

    filtered_df['Sales'] = filtered_df['Quantity'] * filtered_df['UnitPrice']
    sales_trend = filtered_df.groupby('Period')['Sales'].sum().reset_index()
    sales_trend['Period'] = sales_trend['Period'].astype(str)
    sales_trend['Sales'] = sales_trend['Sales'].astype(float)

    # 1) Text summary for terminal
    text = sales_trend.to_string(index=False, max_rows=1000)

    # 2) Raw data for charting (labels + values)
    chart_data = {
        "labels": sales_trend["Period"].tolist(),
        "values": sales_trend["Sales"].tolist()
    }

    return {"text": text, "data": chart_data}


# Tool 2: Top N Products by Revenue
@mcp.tool()
def top_n_products(
    N: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict:
    """
    Get the top N products by revenue, optionally filtered by date range.
    Returns both a text summary and raw data for charting.
    """
    filtered_df = df.copy()

    if start_date:
        filtered_df = filtered_df[filtered_df['InvoiceDate'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df['InvoiceDate'] <= pd.to_datetime(end_date)]

    filtered_df['Revenue'] = filtered_df['Quantity'] * filtered_df['UnitPrice']
    top_n = filtered_df.groupby('Description')['Revenue'].sum().reset_index()
    top_n_sorted = top_n.sort_values(by='Revenue', ascending=False).head(N)

    # 1) Text summary
    text = top_n_sorted.to_string(index=False)

    # 2) Raw data for charting
    chart_data = {
        "labels": top_n_sorted["Description"].tolist(),
        "values": top_n_sorted["Revenue"].tolist()
    }

    return {"text": text, "data": chart_data}


# Tool 3: Category Distribution
@mcp.tool()
def category_distribution() -> dict:
    """
    Auto-group products by category and return sales distribution.
    Returns both a text summary and raw data for charting.
    """
    categories = ["MUG", "LANTERN", "CUSHION"]

    def categorize_product(description):
        for category in categories:
            if re.search(category, description, re.IGNORECASE):
                return category
        return "Other"

    filtered_df = df.copy()
    filtered_df['Category'] = filtered_df['Description'].apply(categorize_product)
    filtered_df['Sales'] = filtered_df['Quantity'] * filtered_df['UnitPrice']
    category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()

    # 1) Text summary
    text = category_sales.to_string(index=False)

    # 2) Raw data for charting
    chart_data = {
        "labels": category_sales["Category"].tolist(),
        "values": category_sales["Sales"].tolist()
    }

    return {"text": text, "data": chart_data}


# Run the server
if __name__ == "__main__":
    mcp.run(transport="stdio")