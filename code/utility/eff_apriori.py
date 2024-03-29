import efficient_apriori

def rules(store_data, support, confidence, max_itemset_cardinality=8):

    #every tuple is a record, the index of the column is also stored
    records = []
    for i in range(0, len(store_data)):
        records.append([(str(store_data.values[i,j]), j) for j in range(0, len(store_data.columns))])

    #find the rules using apriori algorithm
    itemset, rules = efficient_apriori.apriori(records, min_support=support, min_confidence=confidence, max_length=max_itemset_cardinality)
    # print(rules)
    return rules