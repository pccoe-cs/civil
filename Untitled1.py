#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Read CSV as list of lists (transactions)
df_raw = pd.read_csv('transactions.csv', header=None)
transactions = df_raw.values.tolist()

# Remove NaNs from transactions
transactions = [[item for item in transaction if pd.notna(item)] for transaction in transactions]

# Step 2: One-hot encode the data
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print("🧾 Encoded DataFrame:")
print(df_encoded)

# Step 3: Apply Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.6, use_colnames=True)
print("\n✅ Frequent Itemsets:")
print(frequent_itemsets)

# Step 4: Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\n📊 Association Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

