# Windfall — submission (Accel AI Innovate Paris)

## Project name
Windfall

## One-line pitch
Windfall moves small businesses' flexible power use into the hours when electricity is cheap and clean, automatically, and never breaks the cold chain.

## Hosted URL
https://01a162cc-2818-71a1-bd99-489bb9cd522e-8080.eur-1.aiven.app

## Pitch slide (public link)
https://docs.google.com/presentation/d/1DPgYh-Pfllk5EjEpOE3YSJJ2yl-R8xo_HRwF1Pr6xBU/edit?usp=sharing

## Code
https://github.com/Wilivms/windfall

## The data layer in action
Windfall is a team of five agents that run on Aiven Runtime. Every decision they make is built from data stored in Aiven services:

- **Kafka** carries the site's telemetry (`site.telemetry`) and every agent decision (`agent.decisions`). The **Watcher** agent consumes the stream. When a cold-room door is left open or a van comes back late, it replans the rest of the day within seconds.
- **ClickHouse** stores the market history (real French day-ahead prices plus the grid mix, from Energy-Charts), the telemetry and every backtest run. Each night the **Learner** re-fits each cold room's thermal model from this history.
- **PostgreSQL + pgvector** is the source of truth for sites, assets and learned physics. It also holds a vector memory of every past day (a 24-dimension price/carbon profile). The **Forecaster** finds the most similar past days to forecast price and carbon two days ahead, before the market publishes them.
- **Valkey** holds the live site state, plus the `windfall:events` pub/sub bus.
- **OpenSearch** stores a searchable audit trail of every decision. The **Auditor** turns it into the location-based carbon note that a CSRD report needs.
- **Claude** writes the plans and explanations with a structured JSON output. **Keenable** brings in grid news for the Planner. The optimisation itself (dynamic programming under hard limits: −18 °C, 110 kW, vans ready at 06:00) is deterministic and auditable.

## Evidence it works (benchmark vs. a naive site)
Real French market data (Energy-Charts + Open-Meteo), 41 consecutive days, one site: 3 cold rooms and 12 electric vans. The naive baseline is a thermostat plus "charge when plugged in".

| | Naive | Windfall |
|---|---|---|
| Energy cost | €7,180 | **€5,615 (−21.8%)**, about €13,900 saved per year |
| CO₂ (location-based) | 1,704 kg | **1,408 kg (−17.4%)**, 2.6 t per year |
| Peak power | 156 kW | **101 kW** |
| Hours above the 110 kW subscription | 62 h | **0 h** |
| Cold-chain breaches | — | **0** |
| Vans ready at 06:00 | — | **100%** |

- **Learning:** cold-room model error falls from 88% to 0%, and safety overrides fall from 10.8 to 0.2 per day, as the Learner re-fits each night.
- **Forecasting** (2 days ahead, 30 days): price MAE is 40.9 €/MWh with pgvector analog days, against 50.3 for "same as 2 days ago". Carbon MAE is 11.1 against 13.4 g/kWh.
- **Negative prices:** over the period there were 23 hours of negative prices that Windfall used to pre-cool and charge.

## Demo video (3 min) — plan
1. **0:00–0:20 Hook.** "SMEs pay peak prices for power they could use at any time. The grid is full of solar at noon and burns gas at 7 pm."
2. **0:20–0:50 Hero and plan.** Show the dot-matrix of tomorrow's prices (green = the cheapest hours), then the Planner's plan written by Claude, the Keenable news, and the similar past days from pgvector.
3. **0:50–1:40 Live.** Click "Run today live". Telemetry flows through Kafka. A door is left open in Room B and the Watcher replans with Claude's explanation. A van comes back late and the plan is redone. The counters (€, CO₂) update live.
4. **1:40–2:20 Evidence.** The 41-day cost chart (−21.8%), then the time machine on a day with negative prices. Point at the Learner curve and the Forecaster against persistence.
5. **2:20–2:45 Data layer.** The Aiven section: live counts per service, then show the Runtime console with the 5 connected services.
6. **2:45–3:00 Close.** Search the audit trail, then "Draft the CSRD note". Close on: "Windfall. Same power, cheaper hours, cleaner grid."
