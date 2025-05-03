import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import calendar

df = pd.read_csv("Travel details dataset.csv")

df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
df['End date'] = pd.to_datetime(df['End date'], errors='coerce')
df['Travel_month'] = df['Start date'].dt.month
df['Accommodation cost'] = pd.to_numeric(df['Accommodation cost'], errors='coerce')
df['Transportation cost'] = pd.to_numeric(df['Transportation cost'], errors='coerce')
df['Total_cost'] = df['Accommodation cost'] + df['Transportation cost']
df.dropna(subset=['Total_cost', 'Travel_month'], inplace=True)
df['UserID'] = df['Traveler name'].astype('category').cat.codes


user_dest_matrix = df.pivot_table(index='UserID', columns='Destination', values='Total_cost', fill_value=0)
user_similarity = cosine_similarity(user_dest_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_dest_matrix.index, columns=user_dest_matrix.index)


def content_filter(df, preferred_month, max_budget, min_duration=None, max_duration=None, traveler_age=None):
    if isinstance(preferred_month, str):
        preferred_month = list(calendar.month_name).index(preferred_month.title())

    filtered = df.copy()
    filtered = filtered[filtered['Travel_month'] == int(preferred_month)]
    filtered = filtered[filtered['Total_cost'] <= max_budget]

    if min_duration:
        filtered = filtered[filtered['Duration (days)'] >= min_duration]
    if max_duration:
        filtered = filtered[filtered['Duration (days)'] <= max_duration]
    if traveler_age:
        filtered = filtered[np.abs(filtered['Traveler age'] - traveler_age) <= 5]

    return filtered[['Destination', 'Total_cost', 'Duration (days)', 'Travel_month']].drop_duplicates()


def collaborative_filter(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return None, None

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)

    for sim_user_id in similar_users.index:
        user_preferences = user_dest_matrix.loc[sim_user_id]
        recommended = user_preferences[user_preferences > 0].sort_values().head(top_n)

        if not recommended.empty:
            buddy_name_row = df[df['UserID'] == sim_user_id]['Traveler name']
            if not buddy_name_row.empty:
                buddy_name = buddy_name_row.iloc[0]
                return buddy_name, pd.DataFrame(recommended).reset_index().rename(columns={sim_user_id: 'Predicted Cost'})

    return None, None


st.title("🧭 TripWhiz")


traveler_name = st.selectbox("Select your name", sorted(df['Traveler name'].unique()))
user_row = df[df['Traveler name'] == traveler_name]

if user_row.empty:
    st.error("Traveler not found.")
else:
    selected_user_id = user_row['UserID'].iloc[0]
    traveler_age = user_row['Traveler age'].iloc[0] if 'Traveler age' in user_row else None

    st.header("📌 Content-Based Destination Suggestions")

    preferred_month = st.selectbox("Preferred month to travel", list(calendar.month_name)[1:])
    max_budget = st.slider("Maximum total budget (INR)", min_value=1000, max_value=100000, value=20000, step=1000)
    min_days = st.number_input("Minimum duration (days)", value=1, step=1)
    max_days = st.number_input("Maximum duration (days)", value=10, step=1)

    if st.button("Find destinations"):
        content_results = content_filter(
            df, preferred_month, max_budget, min_duration=min_days, max_duration=max_days, traveler_age=traveler_age
        )
        if not content_results.empty:
            st.success("Here are some destinations you might like:")
            st.dataframe(content_results)
        else:
            st.warning("No destinations match your preferences.")

    st.header("👫 Want a Travel Buddy?")
    if st.button("Find me a travel buddy"):
        buddy_name, buddy_recommendations = collaborative_filter(selected_user_id)

        if buddy_name:
            st.success(f"🎒 Travel buddy suggestion: {buddy_name}")
            st.write("Here are some destinations they’ve visited:")
            st.dataframe(buddy_recommendations)
        else:
            st.warning("😕 Sorry, we couldn't find a travel buddy with similar preferences.")
