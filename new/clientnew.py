import asyncio
import re
import json
import os

from contextlib import AsyncExitStack
from dotenv import load_dotenv
import google.generativeai as genai

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

class MCPGeminiClient:
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.model = model
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(self.model)

    async def connect_to_server(self, server_script_path: str = "server.py"):
        params = StdioServerParameters(command="python", args=[server_script_path])
        transport = await self.exit_stack.enter_async_context(stdio_client(params))
        self.stdio, self.write = transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

    async def get_system_prompt(self):
        res = await self.session.call_tool("get_system_prompt", {})
        return res.content[0].text

    async def get_mcp_tools(self):
        tools_res = await self.session.list_tools()
        return [ {"name": t.name, "description": t.description, "parameters": t.inputSchema}
                 for t in tools_res.tools ]

    async def call_gemini_with_retry(self, func, *args, **kwargs):
        for i in range(3):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as e:
                if "429" in str(e) and i < 2:
                    await asyncio.sleep(2 ** i)
                else:
                    raise

    async def process_query(self, query: str) -> str:
        if query.strip().lower() in ["hi","hello"]:
            return "Hello! How can I assist you with sales analysis today?"

        tools = await self.get_mcp_tools()
        system_prompt = await self.get_system_prompt()

        # build helper to map to tool
        helper_prompt = system_prompt + f"""
You are an assistant that maps queries to tool calls.
Available tools: {json.dumps(tools)}
USER QUERY: {query}
Respond ONLY JSON {{"tool":..., "parameters":...}}
"""
        resp = await self.call_gemini_with_retry(
            self.gemini_model.generate_content,
            contents=helper_prompt,
            generation_config={"temperature":0.0}
        )
        text = resp.text.strip()
        # parse JSON
        data = json.loads(re.search(r'\{.*\}', text, re.DOTALL).group(0))
        tool_name  = data["tool"]
        arguments  = data["parameters"]

        # call the tool
        result = await self.session.call_tool(tool_name, arguments)
        payload = json.loads(result.content[0].text)

        # 1) print text summary
        print(payload["text"])

        # 2) ask Gemini to emit ECharts code based on tool_name & data
        chart_data = payload["data"]
        js_helper = f"""
You have three tools:
- dynamic_sales_trend → time-series data
- top_n_products      → ranking data
- category_distribution → categorical data

Current tool: {tool_name}
Data: {json.dumps(chart_data)}

Rules:
- dynamic_sales_trend ⇒ line chart
- top_n_products     ⇒ bar chart
- category_distribution ⇒ pie chart

Generate ONLY JavaScript that:
1. Initializes an ECharts instance on id '{tool_name}-chart'
2. Calls setOption(...) with correct option object
"""
        js_resp = await self.call_gemini_with_retry(
            self.gemini_model.generate_content,
            contents=js_helper,
            generation_config={"temperature":0.0}
        )
        # strip triple backticks if present
        js_code = js_resp.text.replace("```", "").strip()
        if js_code.lower().startswith("javascript"):
            js_code = js_code[len("javascript"):].lstrip()
        # write to static/chart_code.js
        os.makedirs("static", exist_ok=True)
        with open("static/chart_code.js","w") as f:
            f.write(js_code)
        print("👉 Chart code written to static/chart_code.js")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    client = MCPGeminiClient()
    await client.connect_to_server("servernew.py")
    try:
        while True:
            q = input("Enter your query (or 'exit'): ")
            if q.strip().lower()=="exit": 
                break
            await client.process_query(q)
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())