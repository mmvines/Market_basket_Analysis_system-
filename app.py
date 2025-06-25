import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Page config
st.set_page_config(
    page_title="Market Basket Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🛒 Market Basket Analysis using Apriori Algorithm")
st.markdown("Analyze frequently bought items together and uncover hidden shopping patterns.")

# Sidebar: File upload
st.sidebar.title("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose the 'Groceries_dataset.csv' file", type="csv")

# Sidebar: Parameters
st.sidebar.title("⚙️ Apriori Algorithm Settings")
min_support = st.sidebar.slider("🔸 Minimum Support", 0.01, 0.2, 0.06, step=0.01)
min_confidence = st.sidebar.slider("🔸 Minimum Confidence", 0.1, 1.0, 0.4, step=0.05)
min_lift = st.sidebar.slider("🔸 Minimum Lift", 0.5, 5.0, 1.0, step=0.1)

# Main section
if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Show data preview
    with st.expander("📄 Preview Dataset"):
        st.dataframe(data.head(20), use_container_width=True)

    # Top 10 Items Visualization
    st.markdown("### 🥇 Top 10 Most Sold Items")
    top_items = data['itemDescription'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_items.index, y=top_items.values, palette='Set2', ax=ax)
    ax.set_xlabel("Item")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 10 Purchased Items")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Preprocessing
    data['Quantity'] = 1
    basket = data.groupby(['Member_number', 'itemDescription'])['Quantity'] \
                 .sum().unstack().fillna(0)
    basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Frequent itemsets
    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)

    # Association rules
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
    filtered_rules = rules[
        (rules['confidence'] >= min_confidence) & 
        (rules['lift'] >= min_lift)
    ]

    # Show rules
    st.markdown("### 📊 Association Rules")
    if not filtered_rules.empty:
        styled_rules = filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(
            by='lift', ascending=False)
        styled_rules = styled_rules.style \
            .background_gradient(cmap='YlGnBu') \
            .format({'support': '{:.2f}', 'confidence': '{:.2f}', 'lift': '{:.2f}'})
        st.dataframe(styled_rules, use_container_width=True)
    else:
        st.warning("⚠️ No rules found with the selected thresholds. Try adjusting the sliders.")
else:
    st.info("👈 Please upload the 'Groceries_dataset.csv' file to start the analysis.")