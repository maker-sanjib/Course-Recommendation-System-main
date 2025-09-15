import streamlit as st
import pandas as pd
import time
import backend as backend

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Basic webpage setups
st.set_page_config(
    page_title="Course Recommender System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_course_sims():
    return backend.load_course_sims()


@st.cache_data
def load_profile():
    return backend.load_profile()


@st.cache_data
def load_courses():
    return backend.load_courses()


@st.cache_data
def load_courses_genre():
    return backend.load_courses_genre()


@st.cache_data
def load_bow():
    return backend.load_bow()


# Initialize the app by first loading datasets
def init_recommender_app():
    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()
        profile_df = load_profile()
        course_genre_df = load_courses_genre()

    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE'])
    
    if not results.empty:
        st.subheader("Your selected courses: ")
        st.table(results)
    
    return results


def train(model_name, params):
    if model_name in backend.models:
        with st.spinner(f'Training {model_name}...'):
            time.sleep(0.5)
            backend.train(model_name, params)
        st.success(f'{model_name} training completed!')
    else:
        st.error("Invalid Model Selection")


def predict(model_name, user_ids, params):
    if not user_ids:
        st.error("Error: No user IDs provided for prediction.")
        return None

    try:
        with st.spinner('Generating course recommendations...'):
            time.sleep(0.5)
            res = backend.predict(model_name, user_ids, params)
        if res is None or res.empty:
            st.warning("No recommendations found. Try a different input.")
            return None
        st.success('Recommendations generated!')
        return res
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None


# ------ UI ------
st.sidebar.title('Personalized Learning Recommender')

# Initialize the app
selected_courses_df = init_recommender_app()

# Model selection dropdown
st.sidebar.subheader('1. Select Recommendation Model')
model_selection = st.sidebar.selectbox("Select model:", backend.models)

# Hyper-parameters
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters')

# Course similarity model
if model_selection == backend.models[0]:
    params['top_courses'] = st.sidebar.slider('Top courses', 0, 100, 10, 1)
    params['sim_threshold'] = st.sidebar.slider('Course Similarity Threshold %', 0, 100, 50, 10)

# User profile model
elif model_selection == backend.models[1]:
    params['profile_sim_threshold'] = st.sidebar.slider('User Profile Similarity Threshold %', 0, 50, 30, 5)
    temp_user = st.sidebar.text_input("Enter user ID")
    if temp_user.isdigit():
        params['user_id'] = int(temp_user)

# Clustering model
elif model_selection == backend.models[2]:
    params['cluster_no'] = st.sidebar.slider('Number of Clusters', 0, 50, 20, 1)
    temp_user_two = st.sidebar.text_input("Enter user ID to find similar users")
    if temp_user_two.isdigit():
        params['temp_user_two'] = int(temp_user_two)

# Training
st.sidebar.subheader('3. Training')
if st.sidebar.button("Train Model"):
    train(model_selection, params)

# Prediction
st.sidebar.subheader('4. Prediction')
if st.sidebar.button("Recommend New Courses"):
    if model_selection == backend.models[0] and not selected_courses_df.empty:
        new_id = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
        user_ids = [new_id]
    elif model_selection in [backend.models[1], backend.models[2]] and 'user_id' in params:
        user_ids = [params['user_id']]
    else:
        st.warning("Please select courses or enter a valid user ID.")
        user_ids = []

    if user_ids:
        res_df = predict(model_selection, user_ids, params)
        if res_df is not None:
            res_df = res_df[['COURSE_ID', 'SCORE']]
            course_df = load_courses()
            res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)
            st.table(res_df)
