
Use the link below to view the .ipynb file 
nbViewer link : https://nbviewer.org/github/atharvsurve/DCF_ML/blob/main/predDcf.ipynb

# ML-Driven DCF Valuation Engine — A Complete Beginner's Guide

> **What this project does in one sentence:**  
> It uses Machine Learning to predict how much a company is *actually worth* (its "intrinsic value"), compares that to what the stock market says it's worth, and tells you whether to **Buy**, **Hold**, or **Sell**.

---

## Table of Contents

1. [The Big Picture — What Problem Are We Solving?](#1-the-big-picture)
2. [Finance Concepts You Need to Know](#2-finance-concepts-you-need-to-know)
   - [Financial Statements (The Company's Report Card)](#21-financial-statements)
   - [Business Drivers (The Levers That Move a Company)](#22-business-drivers)
   - [DCF Valuation (How Much Is a Company Really Worth?)](#23-dcf-valuation)
   - [Intrinsic Value vs Market Price](#24-intrinsic-value-vs-market-price)
3. [The Pipeline — Step by Step Code Walkthrough](#3-the-pipeline--step-by-step-code-walkthrough)
   - [Step 1 — Load FMP Financial Data](#step-1--load-fmp-financial-data)
   - [Step 2 — Merge Financial Statements](#step-2--merge-financial-statements)
   - [Step 3 — Feature Engineering](#step-3--feature-engineering)
   - [Step 4 — Train ML Models](#step-4--train-ml-models)
   - [Steps 5 & 6 — Forecast Drivers & Financial Statements](#steps-5--6--forecast-drivers--financial-statements)
   - [Step 7 — DCF Valuation](#step-7--dcf-valuation)
   - [Step 8 — Intrinsic Value](#step-8--intrinsic-value)
   - [Steps 9 & 10 — Compare vs Market Price → Buy / Hold / Sell](#steps-9--10--compare-vs-market-price--buy--hold--sell)
4. [Key Assumptions & Limitations](#4-key-assumptions--limitations)
5. [Finance Concepts to Study Next](#5-finance-concepts-to-study-next)

---

## 1. The Big Picture

Imagine you want to buy a house. The seller says it costs ₹1 Crore. But you want to know — **is it really worth ₹1 Crore?** You'd look at the house's rent income, its condition, the neighborhood, etc. and calculate what *you* think it's worth. If your calculation says it's worth ₹1.5 Crore, great deal — buy it! If your calculation says ₹60 Lakhs, it's overpriced — walk away.

**That's exactly what this project does, but for stocks.**

The stock market tells you Apple is trading at $312 per share. But is it *really worth* $312? Our pipeline:
1. Looks at Apple's past 5 years of financial reports (revenue, profits, cash flows, etc.)
2. Uses **Machine Learning** to predict what Apple's finances will look like over the **next 5 years**
3. Calculates the "true" value of Apple using a **Discounted Cash Flow (DCF)** model
4. Compares that true value to the market price
5. Gives you a **Buy / Hold / Sell** recommendation

```
FMP Financial Data           ← "Get the company's report cards"
        ↓
Merge Financial Statements   ← "Combine all reports into one table"
        ↓
Feature Engineering          ← "Calculate the important ratios"
        ↓
Train ML Models              ← "Teach the computer the patterns"
        ↓
Forecast Business Drivers    ← "Predict future ratios"
        ↓
Forecast Financial Statements← "Build future income/cash flow statements"
        ↓
DCF Valuation                ← "Calculate what the company is worth TODAY"
        ↓
Intrinsic Value              ← "Price per share it SHOULD be"
        ↓
Compare vs Market Price      ← "Is the market right or wrong?"
        ↓
Buy / Hold / Sell Signal     ← "What should I do?"
```

---

## 2. Finance Concepts You Need to Know

### 2.1 Financial Statements

Every public company is **legally required** to publish three financial reports every year. Think of them as the company's "report card":

#### 📄 Income Statement — "How much money did you make?"

This shows Revenue (money coming in), Costs (money going out), and Profit (what's left).

```
Revenue (Sales)                    $416 Billion    ← Total money Apple earned
 - Cost of Revenue (COGS)         $221 Billion    ← What it cost to make iPhones
 = Gross Profit                   $195 Billion    ← Profit after production costs
 - Operating Expenses (R&D, SGA)  $ 62 Billion    ← Salaries, research, marketing
 = Operating Income (EBIT)        $133 Billion    ← Profit from core business
 - Taxes                          $ 21 Billion    ← Government's share
 = Net Income                     $112 Billion    ← Final bottom-line profit
```

**Key terms from our code:**
| Term | What It Means | Real-Life Analogy |
|------|---------------|-------------------|
| `revenue` | Total sales | Your salary before any deductions |
| `grossProfit` | Revenue minus production costs | Salary minus raw material costs if you sold things |
| `ebit` | Earnings Before Interest & Tax — profit from core operations | Your business profit before the bank and government take their cut |
| `depreciationAndAmortization` | How much equipment/assets lost value this year | Your car losing ₹1 Lakh in value each year just from aging |
| `incomeBeforeTax` | Profit before the government taxes it | Your income before TDS |
| `incomeTaxExpense` | Taxes paid | TDS / Income tax |
| `netIncome` | Final profit after everything | Your take-home pay |
| `eps` (Earnings Per Share) | Net Income ÷ Number of Shares | If the company was a pizza, how big is each slice |
| `weightedAverageShsOutDil` | Total number of shares that exist | Total number of pizza slices |

#### 📊 Balance Sheet — "What do you own and what do you owe?"

This is a **snapshot** of the company at one point in time. It follows a simple equation:

```
Assets = Liabilities + Shareholders' Equity
(What you OWN = What you OWE + What belongs to shareholders)
```

```
ASSETS (What the company OWNS):
  Cash                           $ 36 Billion     ← Money in the bank
  Short-Term Investments         $ XX Billion     ← Money invested for <1 year
  Receivables                    $ XX Billion     ← Money customers owe you
  Inventory                      $ XX Billion     ← Products sitting in warehouses
  Total Current Assets           $148 Billion     ← Things convertible to cash within 1 year
  Property, Plant & Equipment    $ XX Billion     ← Factories, offices, machines
  Total Assets                   $XXX Billion     ← EVERYTHING the company owns

LIABILITIES (What the company OWES):
  Accounts Payable               $ XX Billion     ← Money owed to suppliers
  Total Current Liabilities      $166 Billion     ← Bills due within 1 year
  Long-Term Debt                 $ XX Billion     ← Loans due after 1 year
  Total Debt                     $112 Billion     ← All borrowings combined
  
EQUITY (What belongs to shareholders):
  Total Stockholders' Equity     $ XX Billion     ← Book value of the company
```

**Key terms from our code:**
| Term | What It Means | Why We Need It |
|------|---------------|----------------|
| `cashAndCashEquivalents` | Money sitting in the bank right now | We ADD this to get equity value |
| `totalCurrentAssets` | Things worth cash within 1 year | Used to calculate Net Working Capital |
| `totalCurrentLiabilities` | Bills due within 1 year | Used to calculate Net Working Capital |
| `totalDebt` | All the money the company borrowed | We SUBTRACT this to get equity value |
| `netDebt` | Total Debt minus Cash | Quick measure of how leveraged a company is |

#### 💰 Cash Flow Statement — "Show me the ACTUAL cash"

The income statement can be misleading (it includes non-cash items). The cash flow statement shows **real cash movement**:

```
Operating Cash Flow          $111 Billion    ← Cash generated from selling iPhones
 - Capital Expenditure       $ 13 Billion    ← Cash spent on building factories/data centers
 = Free Cash Flow            $ 99 Billion    ← Cash left over after reinvesting in the business
```

**Key terms from our code:**
| Term | What It Means | Real-Life Analogy |
|------|---------------|-------------------|
| `operatingCashFlow` | Cash from running the business | Your actual monthly cash income |
| `capitalExpenditure` (Capex) | Cash spent on long-term assets (factories, machines) | Buying a new machine for your shop |
| `freeCashFlow` | Operating Cash Flow minus Capex | Money left after paying all bills AND investing in the future |

> **Why Free Cash Flow matters so much:** It's the cash a company can use to pay dividends, buy back shares, pay down debt, or just save. It's the closest thing to "real profit" because it's actual cash, not accounting profit.

---

### 2.2 Business Drivers

A "business driver" is a **ratio or growth rate** that describes how a company operates. Instead of predicting exact dollar amounts (which vary wildly between Apple and Netflix), we predict **percentages and ratios** — these are more stable and comparable.

Here's each driver our code calculates and what it tells you:

#### Revenue Growth (`rev_growth`)
```python
g['rev_growth'] = rev.pct_change()
# Example: Apple's revenue went from $391B → $416B
# Growth = (416 - 391) / 391 = 6.4%
```
**What it tells you:** Is the company growing? A company growing at 20% is very different from one growing at 2%.

#### Gross Margin (`gross_margin`)
```python
g['gross_margin'] = g['grossProfit'] / rev
# Example: Apple's gross profit is $195B on $416B revenue
# Gross Margin = 195 / 416 = 46.9%
```
**What it tells you:** How much profit does the company keep after paying for the raw materials / production? Apple's 47% means that for every $1 of iPhones sold, it costs only $0.53 to make them. This measures **pricing power** — luxury brands and software companies have high gross margins; grocery stores have low ones.

#### EBIT Margin (`ebit_margin`)
```python
g['ebit_margin'] = g['ebit'] / rev
# Example: Apple's EBIT is $133B on $416B revenue
# EBIT Margin = 133 / 416 = 31.9%
```
**What it tells you:** How profitable is the *core business* after ALL operating costs (salaries, R&D, marketing), but before interest and taxes? This is the most important profitability metric for valuation because it strips out financing decisions (how much debt) and tax jurisdictions.

#### Effective Tax Rate (`eff_tax_rate`)
```python
g['eff_tax_rate'] = (g['incomeTaxExpense'] / g['incomeBeforeTax']).clip(0, 0.40)
```
**What it tells you:** What percentage of profits goes to the government? The US corporate tax rate is 21%, but companies rarely pay exactly that due to deductions, international operations, and tax credits. We clip it between 0% and 40% because loss-making years create weird negative tax rates that would break our model.

#### D&A as % of Revenue (`da_pct_rev`)
```python
g['da_pct_rev'] = g['depreciationAndAmortization'] / rev
```
**What it tells you:** How "asset-heavy" is the company? A factory-heavy company like Intel will have high D&A (lots of expensive equipment losing value). A software company like Meta will have low D&A. We need this because **D&A is a non-cash expense** — the company "lost" value on paper but didn't actually spend cash, so we add it back when calculating Free Cash Flow.

#### Capex as % of Revenue (`capex_pct_rev`)
```python
g['capex_pct_rev'] = g['capitalExpenditure'].abs() / rev
```
**What it tells you:** How much of its revenue does the company reinvest into buying new factories, data centers, equipment? High capex = the company needs to spend a lot to keep growing. Low capex = it can grow without heavy investment (like a software company).

> **Note:** `capitalExpenditure` is **negative** in accounting (it's cash going OUT), so we use `.abs()` to make it positive for our ratio.

#### Net Working Capital as % of Revenue (`nwc_pct_rev`)
```python
g['nwc'] = g['totalCurrentAssets'] - g['totalCurrentLiabilities']
g['nwc_pct_rev'] = g['nwc'] / rev
```
**What it tells you:** How much short-term capital is "tied up" in running the business? If a company's NWC increases, that means it needs more cash tied up in inventory/receivables — that's cash it can't use for other things. **Changes in NWC** affect free cash flow.

---

### 2.3 DCF Valuation

**DCF = Discounted Cash Flow.** This is THE most important valuation method in professional finance. Every investment bank, hedge fund, and private equity firm uses it.

#### The Core Idea: "A dollar today is worth more than a dollar tomorrow"

Would you rather have ₹100 today or ₹100 one year from now? **Today, obviously** — because you could invest it and earn interest. This concept is called the **Time Value of Money**.

So if a company will generate ₹100 in free cash flow next year, that ₹100 isn't worth ₹100 to you today. If you could earn 9% elsewhere (your "discount rate"), then:

```
Present Value = Future Cash Flow / (1 + discount rate)^years

₹100 next year = ₹100 / (1.09)¹ = ₹91.74 today
₹100 in 2 years = ₹100 / (1.09)² = ₹84.17 today
₹100 in 5 years = ₹100 / (1.09)⁵ = ₹64.99 today
```

The further in the future, the less it's worth today.

#### The DCF Formula

A company's value = the sum of ALL future free cash flows, discounted to today:

```
Enterprise Value = Σ (FCFF_t / (1 + WACC)^t)  +  Terminal Value / (1 + WACC)^n

Where:
  FCFF_t        = Free Cash Flow to Firm in year t
  WACC          = Weighted Average Cost of Capital (the discount rate)
  Terminal Value= Value of ALL cash flows beyond year 5 (to infinity)
  n             = Number of forecast years (we use 5)
```

#### What is WACC?

**WACC = Weighted Average Cost of Capital** — it's the minimum return a company must earn on its investments to satisfy both its lenders (banks) and its shareholders.

Think of it as: *"If I invest in this company, what's the opportunity cost? What return could I get elsewhere with similar risk?"*

- A safe, boring company (Walmart) → lower WACC (~7-8%)
- A risky, high-growth company (Tesla) → higher WACC (~10-12%)

**In our code, we use 9% as a reasonable average for large-cap US companies.**

#### What is Terminal Value?

We can only forecast 5 years with any accuracy. But a company doesn't stop existing after year 5! The Terminal Value captures the value of **all cash flows from year 6 to infinity**.

We use the **Gordon Growth Model**:
```
Terminal Value = FCFF_year5 × (1 + g) / (WACC - g)

Where g = perpetual growth rate (we use 2.5%, roughly long-run GDP growth)
```

This assumes the company will grow at a slow, steady rate forever after year 5.

#### From Enterprise Value to Equity Value

Enterprise Value = value of the entire company (debt + equity holders).

But a stock price represents only the **equity** (shareholders' slice). So:

```
Equity Value = Enterprise Value + Cash - Debt
```

Why?
- **+ Cash:** If you bought the whole company, you'd also get its cash pile. That's bonus value.
- **- Debt:** If you bought the whole company, you'd have to pay off its loans. That reduces value.

Finally:
```
Intrinsic Value per Share = Equity Value / Number of Shares Outstanding
```

---

### 2.4 Intrinsic Value vs Market Price

| Scenario | What It Means | Signal |
|----------|--------------|--------|
| Intrinsic Value >> Market Price | The market is **undervaluing** the stock — it's "on sale" | 🟢 **BUY** |
| Intrinsic Value ≈ Market Price | The market is pricing it fairly | 🟡 **HOLD** |
| Intrinsic Value << Market Price | The market is **overvaluing** the stock — it's expensive | 🔴 **SELL** |

We use a **Margin of Safety** (a concept from Benjamin Graham, Warren Buffett's mentor):
- **BUY** only if intrinsic value is **>20% above** market price (i.e., "even if my model is somewhat wrong, I still have a buffer")
- **SELL** only if intrinsic value is **>20% below** market price

---

## 3. The Pipeline — Step by Step Code Walkthrough

### Step 1 — Load FMP Financial Data

```python
income     = pd.read_csv('fmp_dataset/csv/income-statement.csv')
balance    = pd.read_csv('fmp_dataset/csv/balance-sheet-statement.csv')
cashflow   = pd.read_csv('fmp_dataset/csv/cash-flow-statement.csv')
enterprise = pd.read_csv('fmp_dataset/csv/enterprise-values.csv')
quote      = pd.read_csv('fmp_dataset/csv/quote.csv')
dcf_fmp    = pd.read_csv('fmp_dataset/csv/discounted-cash-flow.csv')
```

**What's happening:** We load 6 CSV files that were previously downloaded from the [Financial Modeling Prep (FMP)](https://financialmodelingprep.com/) API using `FMP_api_store.py`. Each CSV contains a different "report card" for 14 big US companies (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX, AMD, INTC, JPM, V, COST, WMT) over 5 years of annual filings.

**Why separate files?** Each financial statement has *different columns*. The Income Statement has revenue and profit columns. The Balance Sheet has asset and debt columns. They can't be combined into one CSV directly — they have to be **merged** first.

---

### Step 2 — Merge Financial Statements

```python
merged = income[inc_cols].merge(balance[bs_cols],  on=['symbol', 'date'], how='inner')
merged = merged.merge(cashflow[cf_cols],           on=['symbol', 'date'], how='inner')
merged = merged.merge(enterprise[ev_cols],         on=['symbol', 'date'], how='inner')
```

**What's happening:** We join four tables together using `(symbol, date)` as the key. For example, the row for "AAPL, 2025-09-27" in the Income Statement gets matched with "AAPL, 2025-09-27" in the Balance Sheet, Cash Flow, and Enterprise Values.

**Why `how='inner'`?** An inner join only keeps rows where a match exists in ALL four tables. If a company has data in the Income Statement but not the Cash Flow statement for some year, that year gets dropped — we need complete data.

**Result:** One clean table where each row = one company-year with revenue, profit, cash, debt, capex, shares, stock price, etc. all in one place.

---

### Step 3 — Feature Engineering

```python
g['rev_growth'] = rev.pct_change()
g['ebit_margin'] = g['ebit'] / rev
# ... etc ...
for col in driver_cols:
    g[f'{col}_lag1'] = g[col].shift(1)
```

**What's happening:** We calculate the 7 business drivers explained in Section 2.2 above for each company. Then we create **lagged versions** — meaning last year's value.

**Why lags?** Our ML model needs to answer: *"Given what the company looked like LAST YEAR, what will it look like THIS YEAR?"*

```
Input (Features):          → Model →     Output (Target):
Last year's rev_growth                    This year's rev_growth
Last year's ebit_margin                   This year's ebit_margin
Last year's tax_rate                      This year's tax_rate
... etc.                                  ... etc.
```

The `.shift(1)` function takes each column and shifts it down by one row. So for AAPL:

| Year | `ebit_margin` | `ebit_margin_lag1` |
|------|--------------|-------------------|
| 2021 | 30.6% | NaN (no previous year) |
| 2022 | 30.9% | 30.6% ← (2021's value) |
| 2023 | 30.7% | 30.9% ← (2022's value) |
| 2024 | 31.5% | 30.7% ← (2023's value) |
| 2025 | 31.9% | 31.5% ← (2024's value) |

---

### Step 4 — Train ML Models

```python
for target in TARGETS:
    rf = RandomForestRegressor(n_estimators=200, max_depth=4, ...)
    rf.fit(X, y)
    models[target] = rf
```

**What's happening:** We train **7 separate Random Forest models** — one for each business driver. Each model learns: "Given a company's characteristics last year, what will this specific driver be this year?"
**Why Random Forest?**
- It's an **ensemble** of 200 decision trees. Each tree looks at the data slightly differently, and the final prediction is the average of all trees. This makes it much more robust than a single decision tree.
- It handles **non-linear relationships** (e.g., "revenue growth tends to slow down as companies get bigger" — a straight line can't capture this, but a tree can).
- It's **robust to outliers** — one weird year for Tesla won't corrupt the whole model.

**Cross-validated R² score:** We report how well each model performs. R² = 1.0 means perfect prediction, R² = 0.0 means the model is no better than just predicting the average. Anything above ~0.3 is decent for financial data (it's inherently noisy).

---

### Steps 5 & 6 — Forecast Drivers & Financial Statements

```python
for yr in range(1, FORECAST_YEARS + 1):
    preds = {t: models[t].predict(feat_input)[0] for t in TARGETS}
    
    proj_rev   = curr_rev * (1 + preds['rev_growth'])
    proj_ebit  = proj_rev * preds['ebit_margin']
    proj_nopat = proj_ebit - proj_tax
    # ... etc ...
    
    fcff = proj_nopat + proj_da - proj_capex - d_nwc
```

**What's happening (the key insight):** This is where ML meets finance. We're doing an **autoregressive forecast** — a fancy way of saying:

1. **Year 1:** Feed the model the *actual* 2025 drivers → it predicts 2026 drivers
2. **Year 2:** Feed the model the *predicted* 2026 drivers → it predicts 2027 drivers
3. **Year 3:** Feed the model the *predicted* 2027 drivers → it predicts 2028 drivers
4. Continue for 5 years total...

For each year, once we have the predicted drivers (percentages), we **reconstruct dollar amounts**:

```
Projected Revenue = Last Year's Revenue × (1 + predicted revenue growth)
Projected EBIT    = Projected Revenue × predicted EBIT margin
Projected Tax     = Projected EBIT × predicted tax rate
NOPAT             = EBIT - Tax  (Net Operating Profit After Tax)
Projected D&A     = Projected Revenue × predicted D&A %
Projected Capex   = Projected Revenue × predicted Capex %
Projected NWC     = Projected Revenue × predicted NWC %
Change in NWC     = This year's NWC - Last year's NWC
```

Finally, we calculate **FCFF (Free Cash Flow to Firm)**:

```
FCFF = NOPAT + D&A − Capex − ΔNWC
```

Why this formula?
- **NOPAT:** Cash profit from operations
- **+ D&A:** We add this back because D&A is a non-cash expense (no actual cash left the building)
- **− Capex:** We subtract this because the company spent real cash on new equipment
- **− ΔNWC:** If working capital increased, cash got "trapped" in inventory/receivables

---

### Step 7 — DCF Valuation

```python
# Discount each year's FCFF to present value
discount_factors = [(1 + WACC) ** yr for yr in range(1, FORECAST_YEARS + 1)]
pv_fcff = sum(f / d for f, d in zip(fcff_list, discount_factors))

# Terminal Value (all cash flows from year 6 to infinity)
terminal_value    = (fcff_list[-1] * (1 + TERMINAL_GROWTH)) / (WACC - TERMINAL_GROWTH)
pv_terminal_value = terminal_value / discount_factors[-1]

enterprise_value  = pv_fcff + pv_terminal_value
```

**What's happening — a worked example:**

Let's say our model predicts Apple's FCFFs as:
| Year | FCFF | Discount Factor (1.09^n) | Present Value |
|------|------|--------------------------|---------------|
| 1 | $110B | 1.090 | $100.9B |
| 2 | $118B | 1.188 | $99.3B |
| 3 | $125B | 1.295 | $96.5B |
| 4 | $131B | 1.412 | $92.8B |
| 5 | $137B | 1.539 | $89.0B |
| **Sum** | | | **$478.5B** |

Terminal Value = ($137B × 1.025) / (0.09 − 0.025) = $2,160B  
PV of Terminal Value = $2,160B / 1.539 = $1,403B

**Enterprise Value = $478.5B + $1,403B = $1,882B**

> Notice how the Terminal Value is often **much larger** than the sum of the 5-year cash flows. This is normal — most of a company's value comes from its long-term future, not just the next 5 years. This is also why the Terminal Growth Rate assumption is so critical.

---

### Step 8 — Intrinsic Value

```python
equity_value = enterprise_value + cash - debt
intrinsic_per_share = equity_value / shares
```

**What's happening:**
```
Enterprise Value    = $1,882B  (value of the whole company)
 + Cash on hand     = $   36B  (bonus: you get this cash if you buy the company)
 - Total Debt       = $  112B  (you'd have to pay off the company's loans)
 = Equity Value     = $1,806B  (value belonging to shareholders)

÷ Shares Outstanding = 15.0B shares

= Intrinsic Value per Share = $120.40
```

---

### Steps 9 & 10 — Compare vs Market Price → Buy / Hold / Sell

```python
final['ML Upside %'] = ((intrinsic - market) / market) * 100

if upside > 20:   return '🟢 BUY'
elif upside < -20: return '🔴 SELL'
else:              return '🟡 HOLD'
```

**What's happening:** We compare our ML-calculated intrinsic value to the live stock price from the `quote.csv` file. We also show FMP's own DCF calculation side-by-side as a benchmark.

The 20% threshold is the **margin of safety** — we only recommend a BUY if we're confident enough that even if our model is off by a bit, it's still a good deal.

---

## 4. Key Assumptions & Limitations

| Assumption | Value | Why |
|-----------|-------|-----|
| WACC | 9% | Reasonable for diversified large-cap US equities |
| Terminal Growth Rate | 2.5% | Approximate long-run nominal GDP growth |
| Forecast Horizon | 5 years | Standard in institutional equity research |
| Margin of Safety | ±20% | Accounts for model uncertainty |

**Limitations to be honest about in interviews:**
1. **Small training set:** Only 14 companies × ~4 usable years = ~56 training samples. More data from more companies and more years would improve the model significantly.
2. **Cross-sectional model:** We train one model across ALL companies. A tech company and a bank operate very differently. Industry-specific models would be more accurate.
3. **Static WACC:** We use 9% for every company. In reality, each company has a different cost of capital based on its risk profile (Beta), capital structure, and credit rating.
4. **No macroeconomic inputs:** The model doesn't consider interest rates, inflation, recessions, or geopolitical events.
5. **Terminal Value dominance:** 60-80% of the enterprise value comes from the Terminal Value, making it extremely sensitive to the growth rate assumption. Changing it from 2.5% to 3.0% can swing the valuation by 15-20%.

---

## 5. Finance Concepts to Study Next

Now that you understand this project, here's what to learn next to go deeper:

| Topic | Why It Matters | Resource |
|-------|---------------|----------|
| **Time Value of Money (TVM)** | Foundation of ALL valuation | Any corporate finance textbook Ch. 1-2 |
| **WACC Calculation** | Learn to calculate it from Beta, Cost of Debt, Capital Structure | Damodaran's website (NYU) |
| **Comparable Company Analysis (Comps)** | Alternative valuation method using P/E, EV/EBITDA ratios | Wall Street Prep |
| **Financial Statement Analysis** | Deep dive into reading 10-K filings | Martin Fridson's book |
| **Beta & CAPM** | How risk translates to required return | Khan Academy or Investopedia |
| **Multiples Valuation** | P/E Ratio, EV/EBITDA, Price/Book — quick valuation shortcuts | Damodaran's "Little Book of Valuation" |
| **Monte Carlo Simulation** | Instead of one DCF, run 10,000 with different assumptions | Python + scipy |
| **Sentiment Analysis (NLP)** | Use earnings call transcripts + ML to predict stock moves | HuggingFace + SEC EDGAR |

---

> **Built with:** Python, Pandas, Scikit-Learn, Financial Modeling Prep API
