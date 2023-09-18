#================================================================
# Final Project: Financial Modeling package
#================================================================

#%% All financial modeling functions

# -----------------------------------------------------------
# Import packages 
import datetime
import pandas as pd
import numpy as np 
from pandas_datareader import data as pdr
import numpy_financial as npf
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------
# [0] Useful functions 

def tab_clean(tab):
    tab.columns = tab.loc[2]
    tab.drop(index = [0,1,2,3], inplace = True)
    tab.drop(columns = np.nan, inplace = True)
    tab.dropna(how = 'all', inplace = True)
    tab.replace('—', 0 , inplace = True)
    tab.set_index('In Millions of USD except Per Share', inplace = True)
    return(tab)

def rewrite_bs(bs_tab):
    fs = pd.DataFrame(columns = bs_tab.columns)

#% Rewrite Balance sheet
    fs.loc['Balance Sheet'] = np.nan
    fs.loc['Cash and marketable securities'] = bs_tab.loc['  + Cash, Cash Equivalents & STI']
    fs.loc['Current assets'] = (bs_tab.loc['  + Accounts & Notes Receiv']
                               + bs_tab.loc['  + Inventories']
                               + bs_tab.loc['  + Other ST Assets'])
    fs.loc['Fixed assets at cost'] = bs_tab.loc['    + Property, Plant & Equip']
    fs.loc['Accumulated depreciation'] = - bs_tab.loc['    - Accumulated Depreciation']
    fs.loc['Net fixed assets'] = bs_tab.loc['  + Property, Plant & Equip, Net']
    fs.loc['Long-term investments'] = bs_tab.loc['  + LT Investments & Receivables']
    fs.loc['Other long-term assets'] = bs_tab.loc['  + Other LT Assets']
    fs.loc['Total assets'] = bs_tab.loc['Total Assets'].iloc[1]
    fs.loc['Current liabilities'] = (bs_tab.loc['  + Payables & Accruals']
                                    + bs_tab.loc['  + Other ST Liabilities'])
    fs.loc['Debt'] = bs_tab.loc['  + ST Debt'] + bs_tab.loc['  + LT Debt']
    fs.loc['Other long-term liabilities'] = bs_tab.loc['  + Other LT Liabilities']
    fs.loc['Total liabilities'] = bs_tab.loc['Total Liabilities']

    fs.loc['Preferred equity'] = bs_tab.loc['  + Preferred Equity and Hybrid Capital']
    fs.loc['Minority interest'] = bs_tab.loc['  + Minority/Non Controlling Interest']
    fs.loc['Common stock'] = bs_tab.loc['  + Share Capital & APIC'] 
    fs.loc['Treasury Stock'] = - bs_tab.loc['  - Treasury Stock'] 
    fs.loc['Accumulated retained earnings'] = bs_tab.loc['  + Retained Earnings']
    fs.loc['Other equity'] = bs_tab.loc['  + Other Equity']  
    fs.loc['Equity'] = (bs_tab.loc['Equity Before Minority Interest'] - 
                        bs_tab.loc['  + Preferred Equity and Hybrid Capital'])
    fs.loc['Total liabilities and equity']  = bs_tab.loc['Total Liabilities & Equity']
    
    return(fs)    

def rewrite_is(is_tab):
    fs = pd.DataFrame(columns = is_tab.columns)
    fs.loc['Income Statement'] = np.nan
    fs.loc['Sales'] = is_tab.loc['Revenue']
    fs.loc['Cost of goods and sold'] = - is_tab.loc['  - Cost of Revenue']
    fs.loc['Depreciation'] = - is_tab.loc['Depreciation Expense']
    fs.loc['Other operating costs'] = -(is_tab.loc['Gross Profit'] 
                                        - is_tab.loc['Depreciation Expense'] 
                                        - is_tab.loc['Operating Income (Loss)'])
    fs.loc['Operating income'] = is_tab.loc['Operating Income (Loss)']
    fs.loc['Interest payments on debt'] = - is_tab.loc['    + Interest Expense']
    fs.loc['Interest earned on cash and marketable securities'] = is_tab.loc['    - Interest Income']
    fs.loc['Other non-operating costs'] = - (is_tab.loc['Operating Income (Loss)']
                                             - is_tab.loc['    + Interest Expense, Net']
                                             - is_tab.loc['Pretax Income (Loss), GAAP'])
    fs.loc['Profit before tax'] = is_tab.loc['Pretax Income (Loss), GAAP']
    fs.loc['Taxes'] = - is_tab.loc['  - Income Tax Expense (Benefit)']
    fs.loc['Other losses and minority interest'] = - (is_tab.loc['Pretax Income (Loss), GAAP']
                                                      - is_tab.loc['  - Income Tax Expense (Benefit)']
                                                      - is_tab.loc['Net Income, GAAP'])
    fs.loc['Profit after tax'] = is_tab.loc['Net Income, GAAP']
    fs.loc['Dividends'] = - is_tab.loc['Total Cash Common Dividends']
    fs.loc['Retained earnings'] = is_tab.loc['Net Income, GAAP'] - is_tab.loc['Total Cash Common Dividends']
    fs = fs.astype(float)
    return(fs)    
# -----------------------------------------------------------
# [1] Enterprise valuation accounting approach


#  LHS
def ev_accounting(firm_bs, year):
    nwc_o = firm_bs.loc['Current assets'] - firm_bs.loc['Current liabilities'] 
    lta = firm_bs.loc['Net fixed assets'] + firm_bs.loc['Long-term investments'] + firm_bs.loc['Other long-term assets']
    ev = nwc_o + lta
    ev = ev[year]
    return ev






# -----------------------------------------------------------
# [2] Enterprise valuation efficient market approach

def ev_effmkt(ticker, firm_bs, year):
    net_debt = firm_bs.loc['Debt'] - firm_bs.loc['Cash and marketable securities']
    ltl = firm_bs.loc['Other long-term liabilities']
    pe = firm_bs.loc['Preferred equity']
    mi = firm_bs.loc['Minority interest']
    #get the market value of the common shares
    firm_quote=pdr.get_quote_yahoo(ticker)
    firm_mktcap=firm_quote.loc[ticker,'marketCap']/1000000
    #EV Efficient market approach
    ev_mkt = net_debt + ltl + pe + mi + firm_mktcap
    ev_mkt=ev_mkt[year]
    return ev_mkt


# -----------------------------------------------------------
# [3] rD: Average cost of existing debt

def rD_avg(firm_bs, firm_is, year):
    net_ip = -(firm_is.loc['Interest payments on debt'] + firm_is.loc['Interest earned on cash and marketable securities'])
    net_debt = firm_bs.loc['Debt'] - firm_bs.loc['Cash and marketable securities']
    avg_net_debt = (net_debt + net_debt.shift(1))/2
    rD_1 = net_ip/avg_net_debt
    rD_1 = rD_1[year]
    return rD_1 



# -----------------------------------------------------------
# [4] rD: Cost of debt based on rating-adjusted yield curve
 
def rD_yldcurve(firm_bond, bonds):
    bonds['YTM'] = bonds['Yld to Mty (Mid)'].astype(float)/100
    bonds['D_Mty'] = pd.to_datetime(bonds['Maturity'])-pd.to_datetime('2020-10-23')
    bonds['Y_Mty'] = bonds['D_Mty']/np.timedelta64(1,'Y')

    bonds['YTM'] = bonds['YTM'].clip(lower = bonds['YTM'].quantile(0.01), 
                                     upper = bonds['YTM'].quantile(0.99))
    bonds['Y_Mty'] = bonds['Y_Mty'].clip(lower = bonds['Y_Mty'].quantile(0.01), 
                                         upper = bonds['Y_Mty'].quantile(0.99))
    bonds['Y_Mty^2'] = bonds['Y_Mty']**2
    bonds['Y_Mty^3'] = bonds['Y_Mty']**3
    polyreg = LinearRegression().fit(X=bonds[['Y_Mty', 'Y_Mty^2', 'Y_Mty^3']], y=bonds['YTM'])
    r_square = polyreg.score(X=bonds[['Y_Mty', 'Y_Mty^2', 'Y_Mty^3']], y=bonds['YTM'])
    intercept = polyreg.intercept_
    coefs = polyreg.coef_
    firm_bond['D_Mty'] = pd.to_datetime(firm_bond['Maturity'])-pd.to_datetime('2020-10-23')
    firm_bond['Y_Mty'] = firm_bond['D_Mty']/np.timedelta64(1,'Y')
    y_mty = firm_bond['Y_Mty'].mean()
    rD_2 = intercept+ coefs[0]*y_mty + coefs[1]*y_mty**2 + coefs[2]*y_mty**3
    out = {'rD':rD_2, 'R-squared':r_square}
    return  out



# -----------------------------------------------------------
# [5] rE: Gordon dividend model

def rE_gordon(ticker, eval_date, est_window):

    firm_quote = pdr.get_quote_yahoo(ticker)
    firm_prc = firm_quote.loc[ticker,'price']

    firm_actions = pdr.get_data_yahoo_actions(ticker, start='1990-01-01', end= str(eval_date))
    firm_div = firm_actions.loc[firm_actions['action']=='DIVIDEND', :]
    
    firm_div.index = firm_div.index.to_period('Q')
    
    yq_now = firm_div.index.max()
    div_now = firm_div.loc[yq_now, 'value']

    n = est_window
    yq_n = yq_now - n*4
   
    div_n = firm_div.loc[yq_n, 'value']
    g_qtr = npf.rate(40,0,div_n, -div_now)
    
    div_now_ann = div_now * 4
    g_ann = (1+g_qtr)**4-1
    
    rE_1 = div_now_ann*(1+g_ann)/firm_prc+g_ann
    
    out = {'rE': rE_1, 'Estimate Window': f'{yq_n} to {yq_now}', 'Div growth': g_ann}
    return out





# -----------------------------------------------------------
# [6] rE: CAPM model

def rE_capm(ticker, eval_date, est_window):
    eval_date = (pd.to_datetime(eval_date)).to_period('M')
    est_start = eval_date - est_window*12
    prices = pdr.get_data_yahoo([ticker,'^GSPC'], start = str(est_start), end = str(eval_date), interval = 'm')    
    prices = prices['Adj Close']
    returns = prices/prices.shift(1) - 1
    returns = returns.dropna()   
    capm = LinearRegression().fit(X = returns[['^GSPC']], y = returns[ticker])
    beta = float(capm.coef_)
    r_sqr = capm.score(X = returns[['^GSPC']], y = returns[ticker])


    erm_start = eval_date - 30*12
    mktprc_30yr = pdr.get_data_yahoo('^GSPC', start = str(erm_start), end = str(eval_date), interval = 'm') 
    mktprc_30yr = mktprc_30yr['Adj Close']
    rM_30yr = mktprc_30yr/mktprc_30yr.shift(1) - 1
    rM_30yr = rM_30yr.dropna()
    e_rM = rM_30yr.mean()
    e_rM_ann = e_rM *12

    rf_30yr = pdr.DataReader('TB3MS', 'fred', start = str(erm_start), end = str(eval_date) )
    e_rf_ann = rf_30yr['TB3MS'].mean()/100

    rE_2 = e_rf_ann + beta *(e_rM_ann-e_rf_ann)
    
    out = {'rE': rE_2, 'CAPM Beta': beta, 'R-squared': r_sqr}
    return out




# -----------------------------------------------------------
# [7] WACC weighted average cost of capital

def wacc(ticker, firm_bs, firm_is, rE, rD, year): 
    firm_quote = pdr.get_quote_yahoo(ticker)
    e = firm_quote.loc[ticker, 'marketCap'] / 1000000
    d = firm_bs.loc['Debt', year] - firm_bs.loc['Cash and marketable securities', year]
    t_c = -firm_is.loc['Taxes']/firm_is.loc['Profit before tax']
    t_c = t_c.mean()
    wacc = e/(e+d) * rE + d/(e+d) * rD * (1-t_c)
    out = {'WACC': wacc, 'equity weight' :e/(e+d), 'debt weight': d/(e+d), 'tax rate': t_c}
    return out 


# -----------------------------------------------------------
# [8] EV: DCF approach based on CSCF

def ev_cscf(ticker, firm_bs, firm_is, firm_cscf, wacc, stg, ltg, year):
    cf_operating = firm_cscf.loc['Cash from Operating Activities']
    cf_investing  = firm_cscf.loc['  + Change in Fixed & Intang']
    fcf_bi = cf_operating + cf_investing
    net_inst = -firm_is.loc['Interest payments on debt']-firm_is.loc['Interest earned on cash and marketable securities']
    t_c = -firm_is.loc['Taxes']/firm_is.loc['Profit before tax']
    fcf = fcf_bi + net_inst * (1-t_c)
    fcf0 = fcf[year]
    cf_fcst = pd.DataFrame(0,index=['fcf','terminalV','total'], columns = range(0,6))
    cf_fcst.loc['fcf'] = npf.fv(stg, cf_fcst.columns, 0, -fcf0)
    cf_fcst.loc['fcf', 0] = 0
    cf_fcst.loc['terminalV', 5] = cf_fcst.loc['fcf',5]*(1+ltg)/(wacc-ltg)
    cf_fcst.loc['total'] = cf_fcst.loc['fcf']+cf_fcst.loc['terminalV']
    ev = npf.npv(wacc, cf_fcst.loc['total'])*(1+wacc)**0.5
    cash = firm_bs.loc['Cash and marketable securities']
    finL = firm_bs.loc['Debt'] + firm_bs.loc['Other long-term liabilities'] + firm_bs.loc['Preferred equity'] + firm_bs.loc['Minority interest'] 
    e_est = ev + cash[year] -finL[year]    
    ## Get shares Outstanding from Yahoo Finance 
    firm_quote = pdr.get_quote_yahoo(ticker)
    shares = firm_quote.loc[ticker, 'sharesOutstanding']/1000000
    pps_est = e_est/shares
    pps_mkt = firm_quote.loc[ticker, 'price']
    out = {'EV':ev , 'Equity value':e_est, 'Per share value':pps_est, 'Actual price per share':pps_mkt, 'Future cash flows':cf_fcst}
    return out



#%% Main code
# -----------------------------------------------------------
# Model input


datapath = 'C:/Users/amuro/Downloads/MRK(3) (1).xlsx'
ticker = 'MRK'
year = 'FY 2019'
eval_date = datetime.date.today()

firm_bs = pd.read_excel(datapath, ticker+'_BS')
firm_bs = rewrite_bs(tab_clean(firm_bs))
firm_is = pd.read_excel(datapath, ticker+'_IS')
firm_is = rewrite_is(tab_clean(firm_is))
firm_cscf = pd.read_excel(datapath, ticker+'_CSCF')
firm_cscf = tab_clean(firm_cscf).dropna()

firm_bond = pd.read_excel(datapath, ticker+'_bond')
bonds = pd.read_excel(datapath,  'Same_rated_bonds')
bonds = (bonds[bonds != '#N/A Field Not Applicable']).dropna()


# -----------------------------------------------------------
# Build models

firm_ev_acc = ev_accounting(firm_bs, year)
firm_ev_mkt = ev_effmkt(ticker, firm_bs, year)
firm_rD1 = rD_avg(firm_bs, firm_is, year)
firm_rD2 = rD_yldcurve(firm_bond, bonds)
firm_rE1 = rE_gordon(ticker, eval_date, 5)
firm_rE2 = rE_capm(ticker, eval_date, 10)
firm_wacc = wacc(ticker, firm_bs, firm_is, (firm_rE1['rE']+firm_rE2['rE'])/2, (firm_rD1+firm_rD2['rD'])/2, year)
firm_ev_dcf = ev_cscf(ticker, firm_bs, firm_is, firm_cscf, firm_wacc['WACC'], 0.06, 0.02, year)


print(f'''Firm: {ticker}, Date: {eval_date}
EV accounting approach: {firm_ev_acc} M
EV efficient market approach: {firm_ev_mkt} M
rD average cost of existing debt: {round((firm_rD1*100),2)}%
rD based on rating-adjusted yield curve: {round((firm_rD2['rD']*100),2)}%
rE gordon dividend model: {round((firm_rE1['rE']*100),2)}%
rE CAPM: {round((firm_rE2['rE']*100),2)}%
WACC: {round((firm_wacc['WACC']*100),2)}%
EV DCF approach: {round(firm_ev_dcf['EV'],1)}
PPS estimated: {round(firm_ev_dcf['Per share value'],2)}
''')




