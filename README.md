# Portfolio Diversification using Monte Carlo Simulation

This project provides a comprehensive analysis of **portfolio diversification** using various **Monte Carlo simulation techniques**. It evaluates different statistical models, simulates future returns, and compares the results with real market data to assess accuracy and robustness.

---

## Folder Structure & Explanation

### **1. Best Portfolio**

**Purpose:**  
Find the **optimal portfolio allocation** by generating multiple portfolios and selecting the one with the **best Sharpe ratio**.

**Process:**

- Generate **N random portfolios** (e.g., 10,000).
- For each portfolio:
  - Run **M Monte Carlo simulations**.
  - Calculate:
    - Expected Return
    - Risk (Standard Deviation)
    - Value at Risk (VaR 95%)
    - Conditional Value at Risk (CVaR 95%)
    - Sharpe Ratio
- Plot **Return vs Risk** (Efficient Frontier).
- Display **Best Portfolio**:
  - Portfolio Weights (Pie Chart)
  - Metrics (Return, Risk, Sharpe, VaR, CVaR)

---

### **2. Monte Carlo Notebooks**

**Purpose:**  
Analyze how different assumptions affect portfolio simulations.

**Combinations Tested:**

| Notebook | Returns | Distribution |
|-----------|---------|--------------|
| Notebook 1 | Daily Returns | Normal Distribution |
| Notebook 2 | Log Returns | Normal Distribution |
| Notebook 3 | Daily Returns | t-Distribution |
| Notebook 4 | Log Returns | t-Distribution |

**Each notebook outputs:**

- Simulated Monte Carlo Paths
- Histogram of Final Portfolio Values
- Key Metrics:
  - **Mean Return**
  - **Standard Deviation (Risk)**
  - **VaR 95%**
  - **CVaR 95%**
  - **Sharpe Ratio**
  - **Sortino Ratio**
  - **Max Drawdown**

---

### **3. Real Life Comparison**

**Purpose:**  
Compare **Monte Carlo simulations vs actual real-world portfolio returns**.

**How it works:**

- Train models on past data (e.g., 2023).
- Simulate future performance (e.g., 2024).
- Compare:
  - **Simulated vs Actual Portfolio Growth**
  - **Histogram of simulated final values vs actual final value**

**Combinations tested:**

| Method | Returns | Distribution |
|---------|---------|--------------|
| Compare 1 | Daily Returns | Normal Distribution |
| Compare 2 | Log Returns | Normal Distribution |
| Compare 3 | Daily Returns | t-Distribution |
| Compare 4 | Log Returns | t-Distribution |

**Metrics:**

- **Average Percentage Error**
- **Maximum Percentage Error**

---

### **4. Real Life Comparison (Driver Code)**

**Purpose:**  
Automate comparison of different methods to find the most realistic simulation.

**What it does:**

- Runs each comparison script **N times** (e.g., 25 times).
- Collects:
  - Mean Average Error
  - Mean Maximum Error
- Identifies the best **distribution and returns combination** for real-life accuracy.

---

## Key Metrics Explained

| Metric | Meaning |
|---|---|
| **Return** | Expected portfolio return |
| **Risk (Std Dev)** | Portfolio volatility |
| **Sharpe Ratio** | Return per unit of risk |
| **VaR 95%** | Value at Risk (95% confidence) |
| **CVaR 95%** | Expected loss beyond VaR |
| **Sortino Ratio** | Return per unit of downside risk |
| **Max Drawdown** | Maximum loss from peak |

---

## Usage

### Install Dependencies

```bash
pip install numpy pandas matplotlib yfinance tabulate
