import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Reproducibility
random.seed(42)
np.random.seed(42)

# Output directory
OUT_DIR = "C:\\Users\\STEGGE\Desktop\\health_ins_poc_full\\files"
os.makedirs(OUT_DIR, exist_ok=True)

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)

# --- DIMENSIONS ---
# Markets
markets = pd.DataFrame({
    'market_id': range(1, 8+1),
    'market_name': ['Ohio', 'Kentucky', 'Indiana', 'Michigan', 'Pennsylvania', 'West Virginia', 'Tennessee', 'Illinois'],
    'region': ['Midwest','South','Midwest','Midwest','Northeast','South','South','Midwest']
})

print("complete")
# Lines of Business and Products
lobs = ['Commercial', 'Medicare', 'Medicaid', 'ACA']
product_rows, product_id_seq = [], 1
for lob in lobs:
    for i in range(1, 4):  # 3 products per LOB
        product_rows.append({
            'product_id': product_id_seq,
            'lob': lob,
            'product_name': f"{lob} Product {i}",
            'metal_level': random.choice(['Bronze','Silver','Gold','Platinum']) if lob == 'ACA' else None,
            'effective_date': START_DATE.date()
        })
        product_id_seq += 1
products = pd.DataFrame(product_rows)
print("complete")
# Plans: 2 per product
plan_rows, plan_id_seq = [], 1
for _, prow in products.iterrows():
    for v in range(1, 3):
        plan_rows.append({
            'plan_id': plan_id_seq,
            'product_id': int(prow['product_id']),
            'plan_name': f"{prow['product_name']} - Plan {v}",
            'actuarial_value': round(random.uniform(0.6, 0.9), 2),
            'csr_variant': random.choice([None, '73', '87', '94']) if prow['lob']=='ACA' else None
        })
        plan_id_seq += 1
plans = pd.DataFrame(plan_rows)
print("complete")
# Employers (groups)
employers = pd.DataFrame({
    'employer_id': range(1, 100+1),
    'employer_name': [f"Employer {i}" for i in range(1, 101)],
    'market_id': np.random.choice(markets['market_id'], 100),
    'segment': np.random.choice(['Small Group', 'Large Group'], 100, p=[0.6, 0.4])
})
print("complete")
# Brokers
brokers = pd.DataFrame({
    'broker_id': range(1, 50+1),
    'broker_name': [f"Broker {i}" for i in range(1, 51)],
    'channel': np.random.choice(['Broker', 'General Agent', 'Direct'], 50, p=[0.7, 0.2, 0.1]),
    'agency': [f"Agency {((i-1)//5)+1}" for i in range(1, 51)]
})
print("complete")
# Channels
channels = pd.DataFrame({
    'channel_id': range(1, 5+1),
    'channel_name': ['Broker', 'Web', 'Call Center', 'Direct', 'General Agent']
})
print("complete")
# Providers
specialties = ['Primary Care', 'Cardiology', 'Oncology', 'Orthopedics', 'Pediatrics', 'OB/GYN', 'Dermatology', 'Psychiatry', 'Pharmacy', 'Hospitalist']
prov_rows = []
for pid in range(1, 500+1):
    spec = random.choice(specialties)
    prov_rows.append({
        'provider_id': pid,
        'npi': f"{1000000000 + pid}",
        'provider_name': f"Provider {pid}",
        'specialty': spec,
        'group_name': f"Group {((pid-1)//20)+1}",
        'market_id': int(np.random.choice(markets['market_id']))
    })
providers = pd.DataFrame(prov_rows)
print("complete")
# Ensure >= 30 Pharmacy providers
pharm_idx = providers[providers['specialty']=='Pharmacy'].shape[0]
needed = 30 - pharm_idx
if needed > 0:
    idxs = providers.sample(n=needed, random_state=42).index
    providers.loc[idxs, 'specialty'] = 'Pharmacy'
print("complete")
# Facilities
fac_types = ['Hospital', 'ASC', 'SNF', 'Clinic']
fac_rows = []
for fid in range(1, 100+1):
    fac_rows.append({
        'facility_id': fid,
        'facility_name': f"Facility {fid}",
        'facility_type': random.choice(fac_types),
        'market_id': int(np.random.choice(markets['market_id']))
    })
facilities = pd.DataFrame(fac_rows)
print("complete")
# Contracts (FFS/Capitation)
contracts = pd.DataFrame({
    'contract_id': range(1, 500+1),
    'provider_id': providers['provider_id'],
    'effective_date': [START_DATE.date() for _ in range(500)],
    'fee_schedule': np.random.choice(['FS-2024-A','FS-2024-B','FS-2025-A'], 500),
    'reimbursement_method': np.random.choice(['FFS','Capitation'], 500, p=[0.85,0.15]),
    'rate': np.round(np.random.uniform(0.7, 1.2, 500), 2)
})
print("complete")
# Value-based contracts (subset)
vbc_subset = providers.sample(n=200, random_state=1)
vbc_contracts = pd.DataFrame({
    'vbc_contract_id': range(1, 200+1),
    'provider_id': vbc_subset['provider_id'].values,
    'measure_set': np.random.choice(['ACO-Primary','Specialty Bundle','PCMH'], 200),
    'upside': np.random.choice([True, False], 200, p=[0.9,0.1]),
    'downside': np.random.choice([True, False], 200, p=[0.4,0.6]),
    'shared_savings_pct': np.round(np.random.uniform(0.1, 0.5, 200), 2)
})
print("complete")
# Networks & membership
networks = pd.DataFrame({
    'network_id': range(1, 10+1),
    'network_name': [f"Network {i}" for i in range(1, 11)],
    'market_id': np.random.choice(markets['market_id'], 10)
})
netprov_rows = []
for _, nrow in networks.iterrows():
    prov_ids = providers.sample(n=80, random_state=int(nrow['network_id']))['provider_id'].tolist()
    for pid in prov_ids:
        netprov_rows.append({'network_id': int(nrow['network_id']), 'provider_id': int(pid)})
network_providers = pd.DataFrame(netprov_rows)
print("complete")
# Drugs
drug_rows = []
for did in range(1, 400+1):
    drug_rows.append({
        'drug_id': did,
        'ndc11': f"{random.randint(10000, 99999)}-{random.randint(1000,9999)}-{random.randint(10,99)}",
        'generic_name': f"GenericDrug{did}",
        'brand_name': f"Brand{did}",
        'specialty_flag': np.random.choice([True, False], p=[0.2, 0.8]),
        'tier': np.random.choice([1,2,3,4], p=[0.3,0.4,0.2,0.1])
    })
drugs = pd.DataFrame(drug_rows)
print("complete")
# Users (IT & Access)
users = pd.DataFrame({
    'user_id': range(1, 300+1),
    'user_name': [f"User {i}" for i in range(1, 301)],
    'department': np.random.choice(['Claims','Finance','Sales','IT','Medical Mgmt','Provider Relations','Compliance','Customer Service'], 300),
    'role': np.random.choice(['Analyst','Manager','Director','Specialist','Engineer','Nurse'], 300)
})
print("complete")
# Campaigns
campaigns = pd.DataFrame({
    'campaign_id': range(1, 50+1),
    'campaign_name': [f"Campaign {i}" for i in range(1, 51)],
    'channel_id': np.random.choice(channels['channel_id'], 50),
    'start_date': pd.to_datetime(np.random.choice(pd.date_range(START_DATE, END_DATE, freq='7D'), 50)).date,
    'end_date': pd.to_datetime(np.random.choice(pd.date_range(START_DATE + timedelta(days=30), END_DATE, freq='7D'), 50)).date
})
print("complete")
# --- MEMBERS ---
first_names = ['Alex','Sam','Jordan','Taylor','Casey','Morgan','Riley','Quinn','Avery','Jamie','Chris','Pat']
last_names = ['Smith','Johnson','Williams','Brown','Jones','Miller','Davis','Garcia','Rodriguez','Wilson']

member_rows = []
for mid in range(1, 2000+1):
    fname = random.choice(first_names)
    lname = random.choice(last_names)
    dob = datetime(1950,1,1) + timedelta(days=random.randint(0, 25000))
    gender = random.choice(['M','F'])
    market_id = int(np.random.choice(markets['market_id']))
    product = products.sample(1, random_state=mid).iloc[0]
    employer_id = int(np.random.choice(employers['employer_id'])) if product['lob'] in ['Commercial','ACA'] and random.random() < 0.8 else None
    plan_id = int(plans[plans['product_id']==product['product_id']].sample(1, random_state=mid)['plan_id'].iloc[0])
    risk_score = round(np.random.normal(1.0, 0.3), 2)
    start = START_DATE + timedelta(days=random.randint(0, 365))
    end = min(start + timedelta(days=random.randint(100, 600)), END_DATE)
    member_rows.append({
        'member_id': mid,
        'subscriber_id': (mid if random.random()<0.7 else random.randint(1, mid)),
        'first_name': fname,
        'last_name': lname,
        'dob': dob.date(),
        'gender': gender,
        'market_id': market_id,
        'product_id': int(product['product_id']),
        'plan_id': plan_id,
        'employer_id': employer_id,
        'risk_score': risk_score,
        'start_date': start.date(),
        'end_date': end.date()
    })
members = pd.DataFrame(member_rows)
print("complete")
# --- FACTS ---
# Enrollment (one per member)
enrollment_rows = []
for idx, m in members.iterrows():
    broker_id = int(np.random.choice(brokers['broker_id'])) if random.random() < 0.7 else None
    channel_id = int(np.random.choice(channels['channel_id']))
    premium = round(np.random.uniform(200, 1200), 2)
    enrollment_rows.append({
        'enrollment_id': idx + 1,
        'member_id': int(m['member_id']),
        'product_id': int(m['product_id']),
        'plan_id': int(m['plan_id']),
        'employer_id': int(m['employer_id']) if not pd.isna(m['employer_id']) else None,
        'broker_id': broker_id,
        'channel_id': channel_id,
        'effective_date': m['start_date'],
        'term_date': m['end_date'],
        'premium_amount': premium
    })
enrollment = pd.DataFrame(enrollment_rows)
print("complete")
# Eligibility (coverage windows)
elig_rows = []
for i, m in members.iterrows():
    pieces = random.randint(1, 2)
    start = pd.to_datetime(m['start_date'])
    for p in range(pieces):
        end = min(start + timedelta(days=random.randint(60, 240)), pd.to_datetime(m['end_date']))
        elig_rows.append({
            'eligibility_id': len(elig_rows)+1,
            'member_id': int(m['member_id']),
            'coverage_start': start.date(),
            'coverage_end': end.date(),
            'status': random.choice(['Active','COBRA','Terminated']) if p==pieces-1 else 'Active'
        })
        start = end + timedelta(days=1)
        if start > pd.to_datetime(m['end_date']):
            break
eligibility = pd.DataFrame(elig_rows)
print("complete")
# Policies (mirror enrollment)
policies = enrollment[['member_id','product_id','plan_id','employer_id','effective_date','term_date']].copy()
policies['policy_id'] = range(1, len(policies)+1)
policies['premium'] = enrollment['premium_amount']
policies = policies[['policy_id','member_id','product_id','plan_id','employer_id','effective_date','term_date','premium']]
print("complete")
# Premium Billing (monthly invoices)
inv_rows = []
for _, pol in policies.iterrows():
    eff = pd.to_datetime(pol['effective_date'])
    term = pd.to_datetime(pol['term_date'])
    months = pd.period_range(eff, term, freq='M')
    for prd in months:
        invoice_date = datetime(prd.year, prd.month, 1)
        due_date = invoice_date + timedelta(days=30)
        amount = float(pol['premium'])
        inv_rows.append({
            'invoice_id': len(inv_rows)+1,
            'policy_id': int(pol['policy_id']),
            'employer_id': int(pol['employer_id']) if not pd.isna(pol['employer_id']) else None,
            'member_id': int(pol['member_id']),
            'invoice_date': invoice_date.date(),
            'due_date': due_date.date(),
            'amount_due': round(amount,2),
            'status': np.random.choice(['Open','Paid','Partial','Written Off'], p=[0.2,0.6,0.15,0.05])
        })
premium_billing = pd.DataFrame(inv_rows)
print("complete")
# Payments
pay_rows = []
for _, inv in premium_billing.iterrows():
    status = inv['status']
    if status in ['Paid','Partial']:
        parts = 1 if status=='Paid' else random.randint(1,2)
        remaining = float(inv['amount_due'])
        for p in range(parts):
            amt = remaining if p==parts-1 else round(remaining * random.uniform(0.3, 0.7), 2)
            remaining = round(remaining - amt, 2)
            pay_rows.append({
                'payment_id': len(pay_rows)+1,
                'invoice_id': int(inv['invoice_id']),
                'amount': amt,
                'payment_date': (pd.to_datetime(inv['invoice_date']) + timedelta(days=random.randint(5,35))).date(),
                'method': np.random.choice(['ACH','Check','Card']),
                'status': 'Posted'
            })
payments = pd.DataFrame(pay_rows)
print("complete")
# AR Aging snapshots
snapshot_dates = [datetime(2024, 6, 30).date(), datetime(2024, 12, 31).date(), datetime(2025, 6, 30).date()]
ar_rows = []
for snap in snapshot_dates:
    for _, inv in premium_billing.sample(n=min(2000, len(premium_billing)), random_state=hash(snap)%2**32).iterrows():
        pays = payments[payments['invoice_id']==inv['invoice_id']]
        paid_by_snap = pays[pays['payment_date']<=snap]['amount'].sum()
        balance = round(float(inv['amount_due']) - float(paid_by_snap), 2)
        if balance <= 0:
            continue
        age_days = (snap - inv['invoice_date']).days
        bucket = '0-30' if age_days<=30 else '31-60' if age_days<=60 else '61-90' if age_days<=90 else '90+'
        ar_rows.append({
            'snapshot_date': snap,
            'customer_type': 'Employer' if not pd.isna(inv['employer_id']) else 'Member',
            'customer_id': int(inv['employer_id']) if not pd.isna(inv['employer_id']) else int(inv['member_id']),
            'invoice_id': int(inv['invoice_id']),
            'aging_bucket': bucket,
            'amount': balance
        })
ar_aging = pd.DataFrame(ar_rows)
print("complete")
# Code sets
cpt_codes = ['99213','99214','99215','99385','93000','12001','20610','36415','80050','70450']
icd10_codes = ['I10','E11.9','J45.909','M54.5','F32.9','N39.0','R51','K21.9','E78.5','Z00.00']
rev_codes = ['0450','0300','0250','0360','0301']
print("complete")
# Claims headers
claim_rows = []
for cid in range(1, 8000+1):
    m = members.sample(1, random_state=cid).iloc[0]
    prov = providers.sample(1, random_state=cid).iloc[0]
    fac = facilities.sample(1, random_state=cid).iloc[0]
    svc_start = START_DATE + timedelta(days=random.randint(0, 700))
    svc_end = svc_start + timedelta(days=random.randint(0, 5))
    billed = round(random.uniform(50, 50000), 2)
    allowed = round(billed * random.uniform(0.3, 0.9), 2)
    paid = round(allowed * random.uniform(0.7, 1.0), 2)
    status = np.random.choice(['Paid','Denied','Pended'], p=[0.85,0.1,0.05])
    claim_rows.append({
        'claim_id': cid,
        'member_id': int(m['member_id']),
        'provider_id': int(prov['provider_id']),
        'facility_id': int(fac['facility_id']),
        'service_start': svc_start.date(),
        'service_end': svc_end.date(),
        'claim_type': np.random.choice(['Professional','Facility']),
        'billed_amt': billed,
        'allowed_amt': allowed,
        'paid_amt': paid if status=='Paid' else 0.0,
        'status': status,
        'received_date': (svc_end + timedelta(days=random.randint(0,10))).date(),
        'adjudicated_date': (svc_end + timedelta(days=random.randint(5,30))).date()
    })
claims = pd.DataFrame(claim_rows)
print("complete")
# Claim Lines
cl_rows, clid = [], 1
for _, ch in claims.iterrows():
    n_lines = random.randint(1, 5)
    for _ in range(n_lines):
        cpt = random.choice(cpt_codes)
        dx = random.choice(icd10_codes)
        rev = random.choice(rev_codes)
        units = random.randint(1, 5)
        lb = round(random.uniform(10, ch['billed_amt']/n_lines), 2)
        la = round(lb * random.uniform(0.3, 0.9), 2)
        lp = la if ch['status']=='Paid' else 0.0
        cl_rows.append({
            'claim_line_id': clid,
            'claim_id': int(ch['claim_id']),
            'cpt_code': cpt,
            'diagnosis_code': dx,
            'rev_code': rev,
            'units': units,
            'line_billed': lb,
            'line_allowed': la,
            'line_paid': lp,
            'denial_reason_code': None if ch['status']=='Paid' else np.random.choice(['CO-50','CO-197','PR-204','CO-45'])
        })
        clid += 1
claim_lines = pd.DataFrame(cl_rows)
print("complete")
# Pharmacy Claims
pharmacy_provider_ids = providers[providers['specialty']=='Pharmacy']['provider_id']
rx_rows = []
for rxid in range(1, 4000+1):
    m = members.sample(1, random_state=rxid).iloc[0]
    pharm = providers.loc[np.random.choice(pharmacy_provider_ids.index)]
    drug = drugs.sample(1, random_state=rxid).iloc[0]
    fill = START_DATE + timedelta(days=random.randint(0, 700))
    qty = round(random.uniform(10, 90), 0)
    days = int(random.choice([30, 60, 90]))
    billed = round(random.uniform(5, 5000), 2)
    allowed = round(billed * random.uniform(0.4, 0.95), 2)
    paid = round(allowed * random.uniform(0.6, 1.0), 2)
    status = np.random.choice(['Paid','Denied'], p=[0.9,0.1])
    rx_rows.append({
        'rx_claim_id': rxid,
        'member_id': int(m['member_id']),
        'pharmacy_provider_id': int(pharm['provider_id']),
        'drug_id': int(drug['drug_id']),
        'fill_date': fill.date(),
        'days_supply': days,
        'quantity': qty,
        'billed': billed,
        'allowed': allowed,
        'paid': paid if status=='Paid' else 0.0,
        'status': status
    })
pharmacy_claims = pd.DataFrame(rx_rows)
print("complete")
# Authorizations
auth_rows = []
for aid in range(1, 3000+1):
    m = members.sample(1, random_state=aid).iloc[0]
    prov = providers.sample(1, random_state=aid).iloc[0]
    req = START_DATE + timedelta(days=random.randint(0, 700))
    dec = req + timedelta(days=random.randint(0, 7))
    decision = np.random.choice(['Approved','Denied','Withdrawn'], p=[0.8,0.15,0.05])
    auth_rows.append({
        'auth_id': aid,
        'member_id': int(m['member_id']),
        'provider_id': int(prov['provider_id']),
        'service_category': np.random.choice(['Imaging','Surgery','Therapy','DME','Hospice']),
        'request_date': req.date(),
        'decision_date': dec.date(),
        'decision': decision,
        'units_approved': int(random.randint(1, 10)) if decision=='Approved' else 0
    })
authorizations = pd.DataFrame(auth_rows)
print("complete")
# Admissions
adm_rows = []
for adid in range(1, 1500+1):
    m = members.sample(1, random_state=adid).iloc[0]
    fac = facilities.sample(1, random_state=adid).iloc[0]
    admit = START_DATE + timedelta(days=random.randint(0, 700))
    discharge = admit + timedelta(days=random.randint(1, 15))
    adm_rows.append({
        'admission_id': adid,
        'member_id': int(m['member_id']),
        'facility_id': int(fac['facility_id']),
        'admit_date': admit.date(),
        'discharge_date': discharge.date(),
        'drg_code': f"{random.randint(100,999)}",
        'readmission_30d': np.random.choice([True, False], p=[0.12, 0.88])
    })
admissions = pd.DataFrame(adm_rows)
print("complete")
# Care Management
care_rows = []
for cid in range(1, 800+1):
    m = members.sample(1, random_state=10000+cid).iloc[0]
    start = START_DATE + timedelta(days=random.randint(0, 650))
    end = start + timedelta(days=random.randint(30, 180))
    care_rows.append({
        'care_case_id': cid,
        'member_id': int(m['member_id']),
        'program': np.random.choice(['Diabetes','CHF','COPD','Asthma','Behavioral Health']),
        'start_date': start.date(),
        'end_date': min(end, END_DATE).date(),
        'risk_tier': np.random.choice(['Low','Medium','High'], p=[0.5,0.35,0.15]),
        'outcome': np.random.choice(['Improved','No Change','Worsened'])
    })
care_management = pd.DataFrame(care_rows)
print("complete")
# Quality Measures
meas_codes = ['HBP','COL','BCS','CDC','W34','FUH']
qm_rows = []
for qid in range(1, 5000+1):
    m = members.sample(1, random_state=20000+qid).iloc[0]
    code = random.choice(meas_codes)
    year = random.choice([2024, 2025])
    denom = 1
    num = np.random.choice([0,1], p=[0.3,0.7])
    qm_rows.append({
        'measure_id': qid,
        'member_id': int(m['member_id']),
        'measure_code': code,
        'measurement_year': year,
        'numerator': num,
        'denominator': denom,
        'compliant': bool(num==1)
    })
quality_measures = pd.DataFrame(qm_rows)
print("complete")
# Provider Scorecards (quarterly)
periods = [(datetime(2024,1,1), datetime(2024,3,31)), (datetime(2024,4,1), datetime(2024,6,30)),
           (datetime(2024,7,1), datetime(2024,9,30)), (datetime(2024,10,1), datetime(2024,12,31)),
           (datetime(2025,1,1), datetime(2025,3,31))]
ps_rows = []
for prov_id in providers['provider_id'].sample(n=300, random_state=99):
    for ps, pe in periods:
        ps_rows.append({
            'provider_id': int(prov_id),
            'period_start': ps.date(),
            'period_end': pe.date(),
            'cost_index': round(np.random.uniform(0.8, 1.2), 2),
            'quality_index': round(np.random.uniform(0.7, 1.3), 2),
            'utilization_index': round(np.random.uniform(0.8, 1.2), 2),
            'total_patients': int(np.random.randint(50, 1000))
        })
provider_scorecards = pd.DataFrame(ps_rows)
print("complete")
# Network Adequacy (by market & specialty)
na_rows = []
for mk in markets['market_id']:
    for spec in specialties:
        prov_count = providers[(providers['market_id']==mk) & (providers['specialty']==spec)].shape[0]
        if prov_count == 0:
            continue
        members_in_mkt = members[members['market_id']==mk].shape[0]
        members_per_provider = round(members_in_mkt / prov_count, 2)
        na_rows.append({
            'market_id': int(mk),
            'specialty': spec,
            'provider_count': int(prov_count),
            'members_per_provider': members_per_provider,
            'avg_distance_miles': round(np.random.uniform(2, 25), 1)
        })
network_adequacy = pd.DataFrame(na_rows)
print("complete")
# GL Accounts and Transactions
gl_accounts = pd.DataFrame({
    'gl_account_id': range(1, 30+1),
    'account_name': ['Cash','Premium Revenue','Claims Expense','Pharmacy Claims Expense','Provider Capitation','Administrative Expense','Reinsurance Recoverable','AR','AP','Unearned Premium','IBNR','Reserves','IT Expense','Salaries','Rent','Depreciation','Interest Income','Tax Expense','Regulatory Fees','Bad Debt','Marketing Expense','Broker Commissions','Quality Incentives','Risk Adjustment Payable','Risk Adjustment Receivable','Stop-Loss Premium','Misc Income','Other Expense','Legal Expense','Consulting Expense'],
    'type': ['Asset','Revenue','Expense','Expense','Expense','Expense','Asset','Asset','Liability','Liability','Liability','Liability','Expense','Expense','Expense','Expense','Revenue','Expense','Expense','Expense','Expense','Expense','Expense','Liability','Asset','Expense','Revenue','Expense','Expense','Expense']
})
gl_txn_rows = []
for tid in range(1, 5000+1):
    acct = gl_accounts.sample(1, random_state=tid).iloc[0]
    amt = round(np.random.uniform(100, 50000), 2)
    sign = -1 if acct['type'] in ['Expense','Liability'] else 1
    gl_txn_rows.append({
        'gl_txn_id': tid,
        'gl_account_id': int(acct['gl_account_id']),
        'txn_date': (START_DATE + timedelta(days=random.randint(0, 700))).date(),
        'amount': sign * amt,
        'source': np.random.choice(['Claims','Premium','Admin','Adjust']),
        'description': f"Auto gen txn {tid}"
    })
gl_transactions = pd.DataFrame(gl_txn_rows)
print("complete")
# Statutory Financials (by quarter and LOB)
quarters = pd.period_range('2024Q1', '2025Q4', freq='Q')
stat_rows = []
for q in quarters:
    for lob in lobs:
        revenue = round(np.random.uniform(5e6, 50e6), 2)
        claims_cost = round(revenue * np.random.uniform(0.7, 0.9), 2)
        admin = round(revenue * np.random.uniform(0.07, 0.15), 2)
        mlr = round(claims_cost / revenue, 4)
        stat_rows.append({'period': str(q), 'lob': lob, 'revenue': revenue, 'claims': claims_cost, 'admin_expense': admin, 'mlr': mlr})
statutory_financials = pd.DataFrame(stat_rows)
print("complete")
# Strategic Initiatives
init_rows = []
for iid in range(1, 25+1):
    start = START_DATE + timedelta(days=random.randint(0, 400))
    target = start + timedelta(days=random.randint(60, 240))
    spent = round(np.random.uniform(10000, 500000), 2)
    budget = spent + round(np.random.uniform(0, 200000), 2)
    init_rows.append({
        'initiative_id': iid, 'name': f"Initiative {iid}", 'owner': random.choice(['COO','CIO','CMO','CFO','Chief Medical Officer']),
        'start_date': start.date(), 'target_date': target.date(),
        'budget': budget, 'spent': spent, 'status': np.random.choice(['On Track','At Risk','Delayed','Completed']),
        'kpi1': round(np.random.uniform(0, 1), 2), 'kpi2': round(np.random.uniform(0, 1), 2)
    })
initiatives = pd.DataFrame(init_rows)
print("complete")
# Regulatory Filings
reg_rows = []
regs = ['CMS','State DOI','HHS','NCQA']
for rid in range(1, 60+1):
    due = START_DATE + timedelta(days=random.randint(30, 700))
    submitted = due - timedelta(days=random.randint(0, 15)) if random.random()<0.8 else None
    reg_rows.append({
        'filing_id': rid, 'regulator': random.choice(regs), 'program': np.random.choice(['Medicare','Medicaid','Commercial','ACA']),
        'due_date': due.date(), 'submitted_date': submitted.date() if submitted else None, 'status': 'Submitted' if submitted else 'Pending'
    })
regulatory_filings = pd.DataFrame(reg_rows)
print("complete")
# Audits
aud_rows = []
for aid in range(1, 60+1):
    opened = START_DATE + timedelta(days=random.randint(0, 650))
    closed = opened + timedelta(days=random.randint(7, 120)) if random.random()<0.7 else None
    aud_rows.append({
        'audit_id': aid, 'area': np.random.choice(['Claims','Eligibility','Provider Data','Security','Pharmacy','Finance']),
        'finding_severity': np.random.choice(['Low','Medium','High','Critical'], p=[0.4,0.35,0.2,0.05]),
        'finding_desc': f"Finding description {aid}",
        'opened_date': opened.date(), 'closed_date': closed.date() if closed else None, 'status': 'Closed' if closed else 'Open'
    })
audits = pd.DataFrame(aud_rows)
print("complete")
# FWA Cases
fwa_rows = []
for fid in range(1, 300+1):
    open_d = START_DATE + timedelta(days=random.randint(0, 650))
    close_d = open_d + timedelta(days=random.randint(7, 240)) if random.random()<0.6 else None
    ident = round(np.random.uniform(500, 200000), 2)
    recov = round(ident * np.random.uniform(0.1, 0.8), 2) if close_d else 0.0
    fwa_rows.append({
        'fwa_id': fid, 'case_open_date': open_d.date(), 'case_close_date': close_d.date() if close_d else None,
        'type': np.random.choice(['Fraud','Waste','Abuse']), 'amount_identified': ident, 'amount_recovered': recov, 'status': 'Closed' if close_d else 'Open'
    })
fwa_cases = pd.DataFrame(fwa_rows)
print("complete")
# Privacy Incidents
pi_rows = []
for pid in range(1, 100+1):
    d = START_DATE + timedelta(days=random.randint(0, 700))
    pi_rows.append({
        'incident_id': pid, 'date': d.date(), 'type': np.random.choice(['PHI Exposure','Ransomware','Lost Device','Unauthorized Access']),
        'records_affected': int(np.random.randint(1, 5000)), 'reported_to_regulator': np.random.choice([True, False], p=[0.2,0.8]),
        'status': np.random.choice(['Investigating','Remediated','Closed'])
    })
privacy_incidents = pd.DataFrame(pi_rows)
print("complete")
# Appeals from denied claims
deny_claims = claims[claims['status']=='Denied']
appeal_rows = []
for _, ch in deny_claims.sample(n=min(800, len(deny_claims)), random_state=123).iterrows():
    rec = pd.to_datetime(ch['received_date'])
    dec = rec + timedelta(days=random.randint(5, 45))
    appeal_rows.append({
        'appeal_id': len(appeal_rows)+1, 'member_id': int(ch['member_id']), 'claim_id': int(ch['claim_id']),
        'level': np.random.choice(['Level 1','Level 2','External']),
        'received_date': rec.date(), 'decision_date': dec.date(),
        'outcome': np.random.choice(['Upheld','Overturned','Partially Overturned'])
    })
appeals = pd.DataFrame(appeal_rows)
print("complete")
# Grievances
griev_rows = []
for gid in range(1, 800+1):
    m = members.sample(1, random_state=30000+gid).iloc[0]
    rec = START_DATE + timedelta(days=random.randint(0, 700))
    res = rec + timedelta(days=random.randint(1, 30))
    griev_rows.append({
        'grievance_id': gid, 'member_id': int(m['member_id']), 'received_date': rec.date(),
        'category': np.random.choice(['Access','Billing','Quality','Customer Service','Benefits']),
        'resolved_date': res.date(), 'outcome': np.random.choice(['Resolved','Withdrawn','Closed-No Issue'])
    })
grievances = pd.DataFrame(griev_rows)
print("complete")
# Customer Service Calls
call_rows = []
for cid in range(1, 4000+1):
    m = members.sample(1, random_state=40000+cid).iloc[0]
    cd = START_DATE + timedelta(days=random.randint(0, 700))
    call_rows.append({
        'call_id': cid, 'member_id': int(m['member_id']), 'reason': np.random.choice(['Benefit Question','Claim Status','ID Card','Billing','Provider Lookup']),
        'call_date': cd.date(), 'handle_time_sec': int(np.random.randint(60, 1800)),
        'resolved': np.random.choice([True, False], p=[0.8,0.2]), 'first_contact_resolution': np.random.choice([True, False], p=[0.7,0.3])
    })
calls = pd.DataFrame(call_rows)
print("complete")
# IT Tickets
apps = ['Claims Sys','Eligibility Sys','Portal','Data Warehouse','Phone System','EHR Interface']
ticket_rows = []
for tid in range(1, 600+1):
    open_d = START_DATE + timedelta(days=random.randint(0, 700))
    res_d = open_d + timedelta(hours=random.randint(1, 240)) if random.random()<0.85 else None
    ticket_rows.append({
        'ticket_id': tid, 'application': random.choice(apps), 'severity': np.random.choice(['P1','P2','P3','P4'], p=[0.05,0.15,0.5,0.3]),
        'opened': open_d, 'resolved': res_d, 'status': 'Resolved' if res_d else 'Open'
    })
tickets = pd.DataFrame(ticket_rows)
print("complete")

# Data Quality Issues
dq_rows = []
for did in range(1, 300+1):
    det = START_DATE + timedelta(days=random.randint(0, 700))
    res = det + timedelta(days=random.randint(1, 60)) if random.random()<0.7 else None
    dq_rows.append({
        'dq_issue_id': did,
        'dataset': np.random.choice(['Claims','Eligibility','Provider','Pharmacy','Finance','Call Center']),
        'issue_type': np.random.choice(['Nulls','Duplicates','Outliers','Mapping Error','Timeliness']),
        'detected_date': det.date(), 'resolved_date': res.date() if res else None,
        'severity': np.random.choice(['Low','Medium','High']), 'record_count': int(np.random.randint(1, 10000))
    })
data_quality_issues = pd.DataFrame(dq_rows)
print("complete")

# Access Logs
access_rows = []
for aid in range(1, 2000+1):
    u = users.sample(1, random_state=aid).iloc[0]
    access_rows.append({
        'access_id': aid, 'user_id': int(u['user_id']),
        'system': np.random.choice(['Claims Sys','Eligibility Sys','GL','DW','Portal']),
        'access_date': (START_DATE + timedelta(days=random.randint(0, 700))).date(),
        'action': np.random.choice(['Read','Write','Export','Admin']), 'success': np.random.choice([True, False], p=[0.98,0.02])
    })
access_log = pd.DataFrame(access_rows)
print("complete")

# Sales Opportunities
opp_rows = []
for oid in range(1, 600+1):
    emp = employers.sample(1, random_state=oid).iloc[0]
    opp_rows.append({
        'opportunity_id': oid, 'employer_id': int(emp['employer_id']),
        'broker_id': int(np.random.choice(brokers['broker_id'])), 'market_id': int(emp['market_id']),
        'stage': np.random.choice(['Prospect','Quoted','Negotiation','Closed Won','Closed Lost'], p=[0.3,0.3,0.2,0.1,0.1]),
        'created_date': (START_DATE + timedelta(days=random.randint(0, 600))).date(),
        'close_date': (START_DATE + timedelta(days=random.randint(100, 700))).date(),
        'status': np.random.choice(['Open','Closed Won','Closed Lost'], p=[0.5,0.3,0.2]),
        'quoted_premium': round(np.random.uniform(10000, 500000), 2), 'lives': int(np.random.randint(5, 2000))
    })
sales_opportunities = pd.DataFrame(opp_rows)
print("complete")

# Quotes
quote_rows = []
for qid in range(1, 900+1):
    opp = sales_opportunities.sample(1, random_state=qid).iloc[0]
    prod = products.sample(1, random_state=qid).iloc[0]
    pl = plans[plans['product_id']==prod['product_id']].sample(1, random_state=qid).iloc[0]
    quote_rows.append({
        'quote_id': qid, 'opportunity_id': int(opp['opportunity_id']),
        'product_id': int(prod['product_id']), 'plan_id': int(pl['plan_id']),
        'quote_date': (START_DATE + timedelta(days=random.randint(0, 650))).date(),
        'quoted_rate': round(np.random.uniform(200, 1200), 2)
    })
quotes = pd.DataFrame(quote_rows)
print("complete")

# Underwriting Decisions
uw_rows = []
for uid in range(1, 400+1):
    emp = employers.sample(1, random_state=uid).iloc[0]
    prod = products.sample(1, random_state=uid).iloc[0]
    plan = plans[plans['product_id']==prod['product_id']].sample(1, random_state=uid).iloc[0]
    uw_rows.append({
        'uw_id': uid, 'employer_id': int(emp['employer_id']),
        'product_id': int(prod['product_id']), 'plan_id': int(plan['plan_id']),
        'decision_date': (START_DATE + timedelta(days=random.randint(0, 700))).date(),
        'rate_change_pct': round(np.random.uniform(-0.1, 0.3), 4),
        'underwritten_pmpm': round(np.random.uniform(250, 900), 2)
    })
underwriting = pd.DataFrame(uw_rows)
print("complete")

# Risk Adjustment
ra_rows = []
for rid in range(1, 3000+1):
    m = members.sample(1, random_state=rid).iloc[0]
    ra_rows.append({
        'ra_id': rid, 'member_id': int(m['member_id']),
        'program': np.random.choice(['Medicare','ACA','Medicaid']),
        'year': np.random.choice([2024, 2025]),
        'raf': round(max(0.1, np.random.normal(1.0, 0.3)), 3),
        'transfer_payment': round(np.random.uniform(-2000, 2000), 2)
    })
risk_adjustment = pd.DataFrame(ra_rows)
print("complete")

# PBM Contracts & Rebates
manufacturers = [f"Mfr {i}" for i in range(1, 30+1)]
pbm_rows = []
for pid in range(1, 60+1):
    start = START_DATE + timedelta(days=random.randint(0, 300))
    end = start + timedelta(days=random.randint(180, 720))
    pbm_rows.append({
        'pbm_contract_id': pid, 'manufacturer': random.choice(manufacturers),
        'start_date': start.date(), 'end_date': min(end, END_DATE).date(),
        'rebate_rate': round(np.random.uniform(0.1, 0.6), 2)
    })
pbm_contracts = pd.DataFrame(pbm_rows)
print("complete")

rebate_rows = []
for rid in range(1, 600+1):
    drug = drugs.sample(1, random_state=rid).iloc[0]
    period = np.random.choice(quarters)
    rebate_rows.append({
        'rebate_id': rid, 'drug_id': int(drug['drug_id']),
        'manufacturer': random.choice(manufacturers),
        'period': str(period), 'amount': round(np.random.uniform(1000, 500000), 2),
        'guaranteed_min': round(np.random.uniform(500, 100000), 2)
    })
rebates = pd.DataFrame(rebate_rows)
print("complete")

# Save helper
def save_csv(df, name):
    fp = os.path.join(OUT_DIR, name)
    df.to_csv(fp, index=False)
    return fp
print("complete")

# Write all CSVs
files = {}
for name, df in [
    ('markets.csv', markets),
    ('products.csv', products),
    ('plans.csv', plans),
    ('employers.csv', employers),
    ('brokers.csv', brokers),
    ('channels.csv', channels),
    ('providers.csv', providers),
    ('facilities.csv', facilities),
    ('contracts.csv', contracts),
    ('vbc_contracts.csv', vbc_contracts),
    ('networks.csv', networks),
    ('network_providers.csv', network_providers),
    ('drugs.csv', drugs),
    ('users.csv', users),
    ('campaigns.csv', campaigns),
    ('members.csv', members),
    ('enrollment.csv', enrollment),
    ('eligibility.csv', eligibility),
    ('policies.csv', policies),
    ('premium_billing.csv', premium_billing),
    ('payments.csv', payments),
    ('ar_aging.csv', ar_aging),
    ('claims.csv', claims),
    ('claim_lines.csv', claim_lines),
    ('pharmacy_claims.csv', pharmacy_claims),
    ('authorizations.csv', authorizations),
    ('admissions.csv', admissions),
    ('care_management.csv', care_management),
    ('quality_measures.csv', quality_measures),
    ('provider_scorecards.csv', provider_scorecards),
    ('network_adequacy.csv', network_adequacy),
    ('gl_accounts.csv', gl_accounts),
    ('gl_transactions.csv', gl_transactions),
    ('statutory_financials.csv', statutory_financials),
    ('initiatives.csv', initiatives),
    ('regulatory_filings.csv', regulatory_filings),
    ('audits.csv', audits),
    ('fwa_cases.csv', fwa_cases),
    ('privacy_incidents.csv', privacy_incidents),
    ('appeals.csv', appeals),
    ('grievances.csv', grievances),
    ('calls.csv', calls),
    ('tickets.csv', tickets),
    ('data_quality_issues.csv', data_quality_issues),
    ('access_log.csv', access_log),
    ('sales_opportunities.csv', sales_opportunities),
    ('quotes.csv', quotes),
    ('underwriting.csv', underwriting),
    ('risk_adjustment.csv', risk_adjustment),
    ('pbm_contracts.csv', pbm_contracts),
    ('rebates.csv', rebates),
]:
    files[name] = save_csv(df, name)
print("all files written complete")
# README with linkages
readme_lines = [
    "# Health Insurer Mock Data",
    "All files are in ./health_insurer_mock",
    "",
    "## Key Linkages",
    "- members.member_id -> claims.member_id, pharmacy_claims.member_id, authorizations.member_id, admissions.member_id, enrollment.member_id, eligibility.member_id, policies.member_id, appeals.member_id, grievances.member_id, calls.member_id, care_management.member_id, quality_measures.member_id, risk_adjustment.member_id",
    "- providers.provider_id -> claims.provider_id, authorizations.provider_id, contracts.provider_id, vbc_contracts.provider_id, pharmacy_claims.pharmacy_provider_id (subset with specialty=Pharmacy), provider_scorecards.provider_id, network_providers.provider_id",
    "- facilities.facility_id -> claims.facility_id, admissions.facility_id",
    "- products.product_id -> plans.product_id, enrollment.product_id, policies.product_id, quotes.product_id, underwriting.product_id",
    "- plans.plan_id -> enrollment.plan_id, policies.plan_id, quotes.plan_id, underwriting.plan_id",
    "- employers.employer_id -> enrollment.employer_id, policies.employer_id, premium_billing.employer_id, sales_opportunities.employer_id, underwriting.employer_id, ar_aging.customer_id (when customer_type=Employer)",
    "- brokers.broker_id -> enrollment.broker_id, sales_opportunities.broker_id",
    "- channels.channel_id -> enrollment.channel_id, campaigns.channel_id",
    "- policies.policy_id -> premium_billing.policy_id",
    "- premium_billing.invoice_id -> payments.invoice_id, ar_aging.invoice_id",
    "- claims.claim_id -> claim_lines.claim_id, appeals.claim_id",
    "- drugs.drug_id -> pharmacy_claims.drug_id, rebates.drug_id",
    "- networks.network_id <-> network_providers.network_id",
    "- markets.market_id -> employers.market_id, providers.market_id, facilities.market_id, sales_opportunities.market_id, members.market_id, networks.market_id, network_adequacy.market_id",
    "- gl_accounts.gl_account_id -> gl_transactions.gl_account_id",
]
with open(os.path.join(OUT_DIR, 'README.txt'), 'w') as f:
    f.write("\n".join(readme_lines))

print(f"CSV files created in ./{OUT_DIR}")