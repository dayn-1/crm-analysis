from sklearn.model_selection import train_test_split

def split_train_test(ds):
    return train_test_split(ds, test_size=0.2, random_state=42)

def keep_full_convos(ds):
    ds = ds['train']['0']
    ds_full_convos = []

    for i in range(1, len(ds)):
        if '<StartOfConversation>' in ds[i]:
            ds_full_convos.append(ds[i-1])

    # Remove the prefix from each conversation
    prefix = "You are a in the role of a Customer. Here is a conversation:\n                    "
    ds_convos_clean = [conv[len(prefix):] if conv.startswith(prefix) else conv for conv in ds_full_convos]
    return ds_convos_clean


def split_cust_sales(text):
    # Split by "Customer:" and "Serviceman:"
    parts = [segment.strip() for segment in text.split("Customer:") if segment]
    split_conversation = []

    for part in parts:
        if "Salesman:" in part:
            split_conversation.extend(part.split("Salesman:"))
        else:
            split_conversation.append(part)

    # Remove empty strings and strip extra whitespace
    split_conversation = [segment.strip() for segment in split_conversation if segment][1:]

    cust = [split_conversation[x] for x in range(0, len(split_conversation), 2)]

    sales = [split_conversation[x] for x in range(1, len(split_conversation), 2)]

    sales[-1] = sales[-1].split('\n')[0]
    
    full_cust = '\n'.join(cust)
    full_sales = '\n'.join(sales)

    full_cust = "Customer says to salesman: " + full_cust
    full_sales = "Salesman says to customer: " + full_sales
    
    return {
        "full": (full_cust, full_sales),
        "split": (cust, sales)
    }