"""Detailed single-question probe through the agent — shows sent query, MCP
success/error, returned-data length, and the full formatted response."""
import sys, json, asyncio, httpx

BASE = "http://localhost:8000"


async def ask(q):
    async with httpx.AsyncClient(timeout=220.0) as c:
        r = await c.post(f"{BASE}/chat", json={"message": q, "user_id": f"dbg_{abs(hash(q))%99999}"})
        d = r.json()
    calls = d.get("mcp_tool_calls", [])
    print("Q:", q)
    for cc in calls:
        pt = cc.get("_parsed_text", "")
        print("  tool   :", cc.get("tool"), "| success:", cc.get("success"), "| latency:", cc.get("latency_ms"))
        print("  sent   :", cc.get("args", {}).get("query"))
        if cc.get("error"):
            print("  error  :", str(cc.get("error"))[:200])
        print("  dataLen:", len(pt), "| data[:240]:", pt[:240].replace("\n", " "))
    print("  RESP   :", d.get("response", "")[:500].replace("\n", " "))
    print()


async def main():
    for q in sys.argv[1:]:
        await ask(q)


if __name__ == "__main__":
    asyncio.run(main())
