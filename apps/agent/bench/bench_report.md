# agent Agent — Benchmark Report

**Date:** 2026-06-03 11:01 UTC
**Queries:** 37
**Success Rate:** 37/37
**Cold Start (first query):** 1343 ms

## Overall Latency
| Metric | Value |
|--------|-------|
| count | 37 ms |
| min | 8548 ms |
| p50 | 14764 ms |
| p75 | 18396 ms |
| p95 | 32894 ms |
| p99 | 43941 ms |
| max | 43941 ms |
| mean | 16532.5 ms |
| stdev | 6894.7 ms |

## Latency by Scenario
| Scenario | Count | p50 | p75 | p95 | Mean | Max |
|----------|-------|-----|-----|-----|------|-----|
| aggregation | 11 | 17358 | 19080 | 23997 | 16696.7 | 23997 |
| multi_filter | 16 | 15016.5 | 18871 | 43941 | 17254.8 | 43941 |
| single_lookup | 10 | 13390.5 | 15808 | 32894 | 15196.1 | 32894 |

## Latency by Stage
| Stage | Count | p50 | p75 | p95 | Mean |
|-------|-------|-----|-----|-----|------|
| enrichment_ms | 37 | 2753 | 3072 | 7110 | 3123.4 |
| guardrail_ms | 37 | 1 | 1 | 3 | 0.9 |
| llm_ms | 37 | 2084 | 3337 | 14097 | 3416.8 |
| mcp_exec_ms | 37 | 7544 | 9419 | 12042 | 7310.3 |
| memory_ms | 37 | 0 | 0 | 0 | 0 |
| parallel_init_ms | 37 | 1 | 1 | 3 | 1.0 |
| tool_selection_ms | 37 | 1698 | 2083 | 2536 | 1765.4 |

## Individual Query Results
| # | ID | Scenario | Total ms | Tool Sel ms | MCP ms | Success |
|---|-----|----------|----------|-------------|--------|---------|
| 1 | Q01 | single_lookup | 10302 | 2882 | 3740 | ✓ |
| 2 | Q02 | multi_filter | 8548 | 1740 | 2631 | ✓ |
| 3 | Q03 | single_lookup | 13131 | 1444 | 5893 | ✓ |
| 4 | Q04 | single_lookup | 15808 | 1672 | 9419 | ✓ |
| 5 | Q05 | single_lookup | 14874 | 2485 | 6055 | ✓ |
| 6 | Q06 | aggregation | 22644 | 2469 | 8686 | ✓ |
| 7 | Q07 | multi_filter | 15527 | 1680 | 9391 | ✓ |
| 8 | Q08 | multi_filter | 14131 | 1179 | 6336 | ✓ |
| 9 | Q09 | multi_filter | 16443 | 1520 | 9492 | ✓ |
| 10 | Q22 | multi_filter | 18871 | 1718 | 2992 | ✓ |
| 11 | Q23 | multi_filter | 12735 | 1095 | 6727 | ✓ |
| 12 | Q24 | multi_filter | 26238 | 1938 | 11522 | ✓ |
| 13 | Q25 | multi_filter | 11108 | 1992 | 3028 | ✓ |
| 14 | Q26 | aggregation | 18396 | 1267 | 12042 | ✓ |
| 15 | Q27 | multi_filter | 10099 | 2536 | 3098 | ✓ |
| 16 | Q28 | single_lookup | 12323 | 1025 | 6946 | ✓ |
| 17 | Q29 | single_lookup | 9423 | 2173 | 2624 | ✓ |
| 18 | Q30 | aggregation | 14282 | 1118 | 7657 | ✓ |
| 19 | Q31 | aggregation | 11713 | 1800 | 5851 | ✓ |
| 20 | Q32 | multi_filter | 14764 | 1345 | 7965 | ✓ |
| 21 | Q33 | multi_filter | 12085 | 1661 | 3144 | ✓ |
| 22 | Q34 | multi_filter | 24256 | 1225 | 17486 | ✓ |
| 23 | Q35 | single_lookup | 32894 | 2159 | 7430 | ✓ |
| 24 | Q38 | multi_filter | 43941 | 1881 | 7979 | ✓ |
| 25 | Q39 | multi_filter | 18374 | 2134 | 8088 | ✓ |
| 26 | Q40 | aggregation | 18593 | 2407 | 9421 | ✓ |
| 27 | Q41 | aggregation | 23997 | 2443 | 9514 | ✓ |
| 28 | Q42 | multi_filter | 13687 | 1342 | 7544 | ✓ |
| 29 | Q43 | multi_filter | 15269 | 1698 | 7132 | ✓ |
| 30 | Q44 | aggregation | 14329 | 951 | 8902 | ✓ |
| 31 | Q45 | single_lookup | 16951 | 1493 | 11562 | ✓ |
| 32 | Q46 | single_lookup | 13650 | 1654 | 8120 | ✓ |
| 33 | Q47 | single_lookup | 12605 | 1579 | 6277 | ✓ |
| 34 | Q48 | aggregation | 13430 | 1910 | 2805 | ✓ |
| 35 | Q49 | aggregation | 9842 | 1562 | 2321 | ✓ |
| 36 | Q50 | aggregation | 17358 | 2083 | 10310 | ✓ |
| 37 | Q51 | aggregation | 19080 | 2061 | 10350 | ✓ |

## Agent Responses (for accuracy comparison)
| ID | Query | Agent Response | Platform Response (paste here) | Accuracy Score |
|----|-------|----------------|-------------------------------|----------------|
| Q01 | What is the risk tolerance and time horizon of investor Deep... | For **Deepak Sharma**:  \| Field \| Value \| \|---\|---\| \| Risk Tolerance \| Moderate \| \| Time Horizon \| Long-term \| | | |
| Q02 | Show all investors with High risk tolerance and Long-term ti... | Found 12 investors with High risk tolerance and Long-term time horizon.  \| Investor Name \| \|---\| \| Neha Sharma \| \| Aaryan Shah \| \| Shanti Banerjee \| \| Shanti Anand \| \| Ritvik Chatterjee \| \| Navya Mitt | | |
| Q03 | List all holdings for investor Deepak Sharma along with curr... | - Holdings for Deepak Sharma could not be retrieved from the available data. The result contains only null values. - Please rephrase with a more specific query if you'd like me to try a narrower looku | | |
| Q04 | Show all SIP transactions for investor Deepak Sharma.... | SIP transactions for **Deepak Sharma**:  \| Investor Name \| Date \| Amount \| Source \| Notes \| \|---\|---:\|---:\|---\|---\| \| Deepak Sharma \| null \| ₹null \| null \| null \|  - No actual SIP transaction details  | | |
| Q05 | What is the sector allocation breakdown for investor Deepak ... | Found 6 sector allocation records for Deepak Sharma.  \| Investor Name \| Sector \| Allocation % \| \|---\|---\|---:\| \| Deepak Sharma \| Government \| 32.68% \| \| Deepak Sharma \| Gold \| 19.84% \| \| Deepak Sharma | | |
| Q06 | How many investors have a liquidity score low?... | Found **15** investors with a **Low** liquidity score.  \| Investor Name \| Liquidity Score \| \|---\|---\| \| Sita Banerjee \| Low \| \| Ritvik Anand \| Low \| \| Aarav Choudhury \| Low \| \| Navya Mukherjee \| Low \| | | |
| Q07 | Which investors have a sector concentration above 80% in any... | Found 3 investors with sector concentration above 80% in a single sector.  \| Investor \| Sector \| Allocation Pct \| \|---\|---\|---:\| \| Vihaan Kumar \| Banking \| 88.08% \| \| Reyansh Mukherjee \| Pharma \| 96.1 | | |
| Q08 | List all investors who prefer the Technology sector.... | Found 16 investors who prefer the Technology sector.  \| Investor ID \| Investor Name \| Sector Focus \| \|---\|---\|---\| \| INV-002 \| Neha Sharma \| Technology, Pharma, Infra \| \| INV-012 \| Rahul Chatterjee \|  | | |
| Q09 | Which investors have a Retirement Planning goal?... | Found 17 investors with the **Retirement Planning** goal.  \| Investor Name \| Investment Goal \| \|---\|---\| \| Pranav Nair \| Retirement Planning \| \| Aditya Mittal \| Retirement Planning \| \| Shreya Mishra \| | | |
| Q22 | Show investor name, their registered goal type, and actual g... | The requested data could not be retrieved because the tool returned: **Context resolution failed**.  Please rephrase the query more specifically, for example by naming a particular investor or goal ty | | |
| Q23 | Which investors have a Wealth Creation registered?... | Could not retrieve any investors with a registered Wealth Creation goal from the available data. If you want, rephrase the query more specifically, for example by investor name or ID. | | |
| Q24 | Which investors have a Wealth Creation or Child Education go... | The data could not be retrieved because the context lookup failed.  Please rephrase your query more specifically, for example by asking for: - investors with the **Wealth Creation** goal - investors w | | |
| Q25 | Show all Target amount and progress percentage for investors... | Found 12 investors with the investment goal **Income Generation**.  \| Investor Name \| Target Amount \| Progress Percentage \| \|---\|---:\|---:\| \| Rajesh Iyer \| ₹0 \| 0% \| \| Krishna Agarwal \| ₹5245803.39 \|  | | |
| Q26 | Top 5 investors by total portfolio value with their risk tol... | Found 5 investors matching the top portfolio value ranking.  \| Investor Name \| Total Portfolio Value \| \|---\|---:\| \| Neha Sharma \| ₹2754592 \| \| Rajesh Iyer \| ₹2513083 \| \| Ritvik Anand \| ₹2487844.82 \| \| | | |
| Q27 | Show all scenario rebalancing records triggered by a Market ... | Found 1 scenario rebalancing record triggered by a Market Crash.  \| Scenario Rebalance Id \| Investor Name \| Triggered Action \| Affected Assets Sectors \| Current Allocation Pct \| New Allocation Pct \| R | | |
| Q28 | What is the overall health score and health status for inves... | \| Metric \| Value \| \|---\|---\| \| Investor \| Neha Sharma \| \| Overall Health Status \| Moderate \| \| Goal Match % \| 36.7% \| \| Risk Score \| High \| \| Liquidity Score \| Moderate \| \| Diversification Score \| Hig | | |
| Q29 | List all investment goals for investor Neha Sharma with targ... | Found 1 investment goal for **Neha Sharma**.  \| Investor Name \| Investment Goal \| Target Amount \| Current Value \| Progress \| Shortfall \| Time To Goal (Months) \| \|---\|---\|---:\|---:\|---:\|---:\|---:\| \| Ne | | |
| Q30 | How many investors have a goal match percentage below 50% (p... | - **1 investor** has a goal match percentage below 50%.  \| Investor Name \| Goal Match % \| \|---\|---:\| \| Neha Sharma \| 36.7% \| | | |
| Q31 | Count of investors whose holdings have returns that are nega... | The data could not be retrieved because the context resolution failed.  Please rephrase the query more specifically, for example by asking for: - loss-making holdings for a specific investor, or - inv | | |
| Q32 | List all large transactions (amount above 200,000) across al... | Found 3 large transactions above ₹200,000 in 2024 or later.  \| Investor \| Date \| Type \| Amount \| Source \| Notes \| \|---\|---\|---:\|---:\|---\|---\| \| Nisha Singh \| 2024-05-18 \| Lump Sum \| ₹250000 \| Mindspac | | |
| Q33 | For each investor, show risk tolerance alongside overall hea... | Found 12 investors with **High** risk tolerance and **Long-term** horizon.  \| Investor Name \| Risk Tolerance \| Investment Horizon \| Overall Health Score \| \|---\|---\|---\|---\| \| Neha Sharma \| High \| Long | | |
| Q34 | List investors with a Market Crash scenario trigger and show... | Found 0 investors with a **Market Crash** scenario trigger in the available data.  \| Investor Name \| Current Goal Match Percentage \| \|---\|---\| \| No matching investors found in the provided data \| — \|  | | |
| Q35 | For investor Deepak Sharma, show all holdings with their cor... | Found 90 holdings records for **Deepak Sharma**.  \| Investor Name \| Investment Name \| Investment Type \| Sector \| Current Value \| Allocation Pct \| \|---\|---\|---\|---\|---:\|---:\| \| Deepak Sharma \| Gov Bond | | |
| Q38 | Show investor name, sector, and allocation percentage for al... | Found 100 high risk tolerance allocation records.  \| Investor Name \| Sector \| Allocation Percentage \| \|---\|---\|---:\| \| Neha Sharma \| Technology \| 21.11% \| \| Neha Sharma \| Pharma \| 13.46% \| \| Neha Shar | | |
| Q39 | List all investment goals that are behind target (progress b... | Found 11 investment goals that are behind target (progress below 20%).  \| Investor \| Investment Goal \| Target Amount \| Current Value \| Progress % \| Shortfall \| Time To Goal Months \| \|---\|---\|---:\|---: | | |
| Q40 | For each investor, show total portfolio value, total SIP con... | Found 7 investors.  \| Investor \| Total Portfolio Value \| Total SIP Contributed \| Overall Health Score \| \|---\|---:\|---:\|---\| \| Deepak Sharma \| ₹73392438 \| — \| — \| \| Krishna Agarwal \| ₹1427144.6 \| — \| — | | |
| Q41 | Show total cash inflows vs SWP withdrawals per investor, alo... | I found **93 records**, but the data provided contains only **Investor Name / ID** and **Investment Goal**. It does **not** include any cash inflow or SWP withdrawal amounts, so those totals cannot be | | |
| Q42 | Which investors have a sector overweight (>90% in any sector... | Found 1 investor with a sector overweight above 90% in a sector.  \| Investor \| Sector \| Allocation Pct \| Total Investment \| \|---\|---\|---:\|---:\| \| Reyansh Mukherjee \| Pharma \| 96.15% \| ₹946146 \| | | |
| Q43 | List all investors with an Emergency Fund goal.... | Found **14 investors** with an **Emergency Fund** goal.  \| Investor ID \| Investor Name \| Investment Goal \| \|---\|---\|---\| \| INV-008 \| Sita Banerjee \| Emergency Fund \| \| INV-014 \| Sara Joshi \| Emergency | | |
| Q44 | How many investors are in each risk tolerance group?... | Found 100 investors. Here’s the count in each risk tolerance group:  \| Risk Tolerance \| Investor Count \| \|---\|---:\| \| High \| 36 \| \| Moderate \| 33 \| \| Low \| 31 \| | | |
| Q45 | What is the total current portfolio value for investor Deepa... | \| Metric \| Value \| \|---\|---:\| \| Investor Name \| Deepak Sharma \| \| Total Current Portfolio Value \| ₹23,67,498 \| | | |
| Q46 | What is the total SIP contribution amount for investor Deepa... | - Total SIP contribution amount for **Deepak Sharma** is **₹445000**. | | |
| Q47 | What is the total transaction amount per transaction type fo... | Here is the total transaction amount by transaction type for **Neha Sharma**:  \| Transaction Type \| Total Amount \| \|---\|---:\| \| Interest \| ₹421326 \| \| Lump Sum \| ₹3300000 \| \| Dividend \| ₹165636 \| \| SW | | |
| Q48 | Which sector has the highest total investment across all inv... | - The sector with the highest total investment across all investors is **Real Estate** at **₹29384043.2**. | | |
| Q49 | What is the average goal match percentage across all investo... | - The average goal match percentage across all investors is **39.4949999046326%**. | | |
| Q50 | How many rebalancing events were triggered per scenario type... | - The data could not be retrieved because the tool call failed: **Context resolution failed**. - Please rephrase the query more specifically, for example by including an **investor scope** or a narrow | | |
| Q51 | What is the average annual return and volatility across all ... | Found 7 goal types with average annual return and volatility grouped by investment goal.  \| Investment Goal \| Avg Annual Return \| Avg Volatility \| \|---\|---:\|---:\| \| Wealth Preservation \| 12.0753844334 | | |