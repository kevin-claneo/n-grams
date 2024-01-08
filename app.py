# inspired by the GSC-Connector from Lee Foot https://github.com/searchsolved/search-solved-public-seo/tree/main/portfolio/gsc-connector

# Standard library imports
import datetime
import base64

# Related third-party imports
import streamlit as st
from streamlit_elements import Elements
from streamlit_tags import st_tags
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import pandas as pd
import searchconsole

from collections import Counter
import re
import plotly.express as px
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Configuration: Set to True if running locally, False if running on Streamlit Cloud
# IS_LOCAL = True
IS_LOCAL = False

# Constants
SEARCH_TYPES = ["web", "image", "video", "news", "discover", "googleNews"]
DATE_RANGE_OPTIONS = [
    "Last 7 Days",
    "Last 30 Days",
    "Last 3 Months",
    "Last 6 Months",
    "Last 12 Months",
    "Last 16 Months",
]
DEVICE_OPTIONS = ["All Devices", "desktop", "mobile", "tablet"]
BASE_DIMENSIONS = ["page", "query", "country", "date"]
MAX_ROWS = 250_000
DF_PREVIEW_ROWS = 100


# -------------
# Streamlit App Configuration
# -------------

def setup_streamlit():
    """
    Configures Streamlit's page settings and displays the app title and markdown information.
    Sets the page layout, title, and markdown content with links and app description.
    """
    st.set_page_config(
    page_title="Topical Authority with N-grams - Kevin (Claneo)",
    page_icon=":weight_lifter:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/kirchhoff-kevin/',
        'About': "This is an app for checking your topical authority! Adapted from Lee Foot's GSC-connector check out his apps: https://leefoot.co.uk"
    }
    )
    st.image("https://www.claneo.com/wp-content/uploads/Element-4.svg", width=600, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.caption(":point_right: Join Claneo and support exciting clients as part of the Consulting team") 
    st.caption(':bulb: Make sure to mention that *Kevin* brought this job posting to your attention')
    st.link_button("Learn More", "https://www.claneo.com/en/career/#:~:text=Consulting")
    st.title("Check the topical authority of a GSC property")
    st.divider()


def init_session_state():
    """
    Initialises or updates the Streamlit session state variables for property selection,
    search type, date range, dimensions, and device type.
    """
def init_session_state():
    """
    Initializes or updates the Streamlit session state variables.
    """
    if 'selected_property' not in st.session_state:
        st.session_state.selected_property = None
    if 'selected_search_type' not in st.session_state:
        st.session_state.selected_search_type = 'web'
    if 'selected_date_range' not in st.session_state:
        st.session_state.selected_date_range = 'Last 7 Days'
    if 'selected_dimensions' not in st.session_state:
        st.session_state.selected_dimensions = ['query','page']
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = 'All Devices'
    if 'selected_max_position' not in st.session_state:
        st.session_state.selected_max_position = 100
    if 'selected_min_clicks' not in st.session_state:
        st.session_state.selected_min_clicks = 1


# -------------
# Google Authentication Functions
# -------------

def load_config():
    """
    Loads the Google API client configuration from Streamlit secrets.
    Returns a dictionary with the client configuration for OAuth.
    """
    client_config = {
        "installed": {
            "client_id": str(st.secrets["installed"]["client_id"]),
            "client_secret": str(st.secrets["installed"]["client_secret"]),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://accounts.google.com/o/oauth2/token",
            "redirect_uris": (
                ["http://localhost:8501"]
                if IS_LOCAL
                else [str(st.secrets["installed"]["redirect_uris"][0])]
            ),
        }
    }
    return client_config


def init_oauth_flow(client_config):
    """
    Initialises the OAuth flow for Google API authentication using the client configuration.
    Sets the necessary scopes and returns the configured Flow object.
    """
    scopes = ["https://www.googleapis.com/auth/webmasters"]
    return Flow.from_client_config(
        client_config,
        scopes=scopes,
        redirect_uri=client_config["installed"]["redirect_uris"][0],
    )


def google_auth(client_config):
    """
    Starts the Google authentication process using OAuth.
    Generates and returns the OAuth flow and the authentication URL.
    """
    flow = init_oauth_flow(client_config)
    auth_url, _ = flow.authorization_url(prompt="consent")
    return flow, auth_url


def auth_search_console(client_config, credentials):
    """
    Authenticates the user with the Google Search Console API using provided credentials.
    Returns an authenticated searchconsole client.
    """
    token = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
        "id_token": getattr(credentials, "id_token", None),
    }
    return searchconsole.authenticate(client_config=client_config, credentials=token)


# -------------
# Data Fetching Functions
# -------------

def list_gsc_properties(credentials):
    """
    Lists all Google Search Console properties accessible with the given credentials.
    Returns a list of property URLs or a message if no properties are found.
    """
    service = build('webmasters', 'v3', credentials=credentials)
    site_list = service.sites().list().execute()
    return [site['siteUrl'] for site in site_list.get('siteEntry', [])] or ["No properties found"]


def fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, max_position, min_clicks, brand_keywords, device_type=None):
    """
    Fetches Google Search Console data for a specified property, date range, dimensions, and device type.
    Filters the data to include only queries with an average position equal to or lower than max_position.
    """
    query = webproperty.query.range(start_date, end_date).search_type(search_type).dimension(*dimensions)

    if 'device' in dimensions and device_type and device_type != 'All Devices':
        query = query.filter('device', 'equals', device_type.lower())

    try:
        df = query.limit(MAX_ROWS).get().to_dataframe()
        if 'clicks' in df.columns and 'position' in df.columns:
            df = df[df['clicks'] >= min_clicks]
            df = df[df['position'] <= max_position]
        else:
            print("Columns 'clicks' or 'position' not in DataFrame")

        print("Data after filtering:", df.head())
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()
        
    except Exception as e:
        show_error(e)
        return pd.DataFrame()


def fetch_data_loading(webproperty, search_type, start_date, end_date, dimensions, max_position, min_clicks, brand_keywords, device_type=None):
    """
    Fetches Google Search Console data with a loading indicator. Utilises 'fetch_gsc_data' for data retrieval.
    Returns the fetched data as a DataFrame.
    """
    with st.spinner('Fetching data...'):
        return fetch_gsc_data(webproperty, search_type, start_date, end_date, dimensions, max_position, min_clicks, device_type, brand_keywords)


# -------------
# Utility Functions
# -------------

def update_dimensions(selected_search_type):
    """
    Updates and returns the list of dimensions based on the selected search type.
    Adds 'device' to dimensions if the search type requires it.
    """
    return BASE_DIMENSIONS + ['device'] if selected_search_type in SEARCH_TYPES else BASE_DIMENSIONS


def calc_date_range(selection):
    """
    Calculates the date range based on the selected range option.
    Returns the start and end dates for the specified range.
    """
    range_map = {
        'Last 7 Days': 7,
        'Last 30 Days': 30,
        'Last 3 Months': 90,
        'Last 6 Months': 180,
        'Last 12 Months': 365,
        'Last 16 Months': 480
    }
    today = datetime.date.today()
    return today - datetime.timedelta(days=range_map.get(selection, 0)), today


def show_error(e):
    """
    Displays an error message in the Streamlit app.
    Formats and shows the provided error 'e'.
    """
    st.error(f"An error occurred: {e}")


def property_change():
    """
    Updates the 'selected_property' in the Streamlit session state.
    Triggered on change of the property selection.
    """
    st.session_state.selected_property = st.session_state['selected_property_selector']


# -------------
# File & Download Operations
# -------------

def show_dataframe(report):
    """
    Shows a preview of the first 100 rows of the report DataFrame in an expandable section.
    """
    with st.expander("Preview the First 100 Rows"):
        st.dataframe(report.head(DF_PREVIEW_ROWS))


def download_csv_link(report):
    """
    Generates and displays a download link for the report DataFrame in CSV format.
    """
    def to_csv(df):
        return df.to_csv(index=False, encoding='utf-8-sig')

    csv = to_csv(report)
    b64_csv = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64_csv}" download="search_console_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)


# -------------
# Streamlit UI Components
# -------------

def show_google_sign_in(auth_url):
    """
    Displays the Google sign-in button and authentication URL in the Streamlit sidebar.
    """
    with st.sidebar:
        if st.button("Sign in with Google"):
            # Open the authentication URL
            st.write('Please click the link below to sign in:')
            st.markdown(f'[Google Sign-In]({auth_url})', unsafe_allow_html=True)


def show_property_selector(properties, account):
    """
    Displays a dropdown selector for Google Search Console properties.
    Returns the selected property's webproperty object.
    """
    selected_property = st.selectbox(
        "Select a Search Console Property:",
        properties,
        index=properties.index(
            st.session_state.selected_property) if st.session_state.selected_property in properties else 0,
        key='selected_property_selector',
        on_change=property_change
    )
    return account[selected_property]


def show_search_type_selector():
    """
    Displays a dropdown selector for choosing the search type.
    Returns the selected search type.
    """
    return st.selectbox(
        "Select Search Type:",
        SEARCH_TYPES,
        index=SEARCH_TYPES.index(st.session_state.selected_search_type),
        key='search_type_selector'
    )


def show_date_range_selector():
    """
    Displays a dropdown selector for choosing the date range.
    Returns the selected date range option.
    """
    return st.selectbox(
        "Select Date Range:",
        DATE_RANGE_OPTIONS,
        index=DATE_RANGE_OPTIONS.index(st.session_state.selected_date_range),
        key='date_range_selector'
    )


def show_dimensions_selector(search_type):
    """
    Displays a multi-select box for choosing dimensions based on the selected search type.
    Returns the selected dimensions.
    """
    available_dimensions = update_dimensions(search_type)
    return st.multiselect(
        "Select Dimensions:",
        available_dimensions,
        default=st.session_state.selected_dimensions,
        key='dimensions_selector'
    )

def show_max_position_selector():
    """
    Displays a slider for choosing the maximum position for queries.
    Returns the selected maximum position.
    """
    max_position = st.slider(
        "Select Maximum Position for Queries:",
        min_value=1, 
        max_value=100, 
        value=(st.session_state.selected_max_position)
    )
    st.session_state.selected_max_position = max_position
    return max_position

def show_min_clicks_input():
    """
    Displays a number input for specifying the minimum number of clicks.
    Updates the session state with the selected value.
    """

    min_clicks = st.number_input("Minimum Number of Clicks:", min_value=0, value=st.session_state.selected_min_clicks)
    st.session_state.selected_min_clicks = min_clicks
    return min_clicks



fetched_data = None

def show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions, max_position, min_clicks, brand_keywords):
    """
    Displays a button to fetch data based on selected parameters.
    Shows the report DataFrame upon successful data fetching.
    """
    if st.button("Fetch Data"):
        report = fetch_data_loading(webproperty, search_type, start_date, end_date, selected_dimensions, max_position, min_clicks, brand_keywords)
        for brand_keyword in brand_keywords:
                        report = report[~report['query'].str.contains(brand_keyword, case=False, na=False)]
        
        if report is not None:
            st.session_state.fetched_data = report  # Store in session state
            
def process_ngrams(df, numGrams, minOccurrences=1):
    # Ensure 'clicks' column is of integer type
    df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)

    # Define stopwords and regex for allowed characters (including German characters)
    stop_words = set(stopwords.words('english')) | set(stopwords.words('german'))
    re_allowed_chars = re.compile("[^A-Za-z0-9 '’äöüßÄÖÜ]+")

    # Clean and tokenize the queries
    def clean_and_tokenize(query):
        query = re_allowed_chars.sub('', query.lower())
        words = query.split()
        words = [word for word in words if word not in stop_words]
        return [tuple(words[i:i + numGrams]) for i in range(len(words) - numGrams + 1)]

    # Apply the function to the DataFrame
    df['ngrams'] = df['query'].apply(clean_and_tokenize)

    # Flatten the list of ngrams and count occurrences
    ngrams_flat = [ngram for sublist in df['ngrams'] for ngram in sublist]
    ngram_counts = Counter(ngrams_flat)


    # Sum clicks for each ngram
    ngram_clicks = {}
    ngram_pages = {}

    calculate_pages = 'page' in df.columns
     # Process each row for ngrams, clicks, and pages (if applicable)
    for index, row in df.iterrows():
        for ngram in row['ngrams']:
            if ngram in ngram_counts and ngram_counts[ngram] >= minOccurrences:
                ngram_clicks[ngram] = ngram_clicks.get(ngram, 0) + row['clicks']
                if calculate_pages:
                    ngram_pages[ngram] = ngram_pages.get(ngram, set()).union({row['page']})

    # Summarize results
    final_data = []
    for ngram, count in ngram_counts.items():
        if count >= minOccurrences:
            clicks = ngram_clicks.get(ngram, 0)
            ngram_str = ' '.join(ngram)
            if calculate_pages:
                pages_count = len(ngram_pages[ngram])
                final_data.append([ngram_str, count, clicks, pages_count])
            else:
                final_data.append([ngram_str, count, clicks])

    # Define columns based on whether 'page' is in the DataFrame
    columns = ['Ngram', 'Occurrences', 'Total Clicks']
    if calculate_pages:
        columns.append('Unique Pages')

    final_df = pd.DataFrame(final_data, columns=columns)
    return final_df.sort_values(by='Total Clicks', ascending=False)

# -------------
# Main Streamlit App Function
# -------------

# Main Streamlit App Function
def main():
    """
    The main function for the Streamlit application.
    Handles the app setup, authentication, UI components, and data fetching logic.
    """
    setup_streamlit()
    client_config = load_config()
    st.session_state.auth_flow, st.session_state.auth_url = google_auth(client_config)

    query_params = st.experimental_get_query_params()
    auth_code = query_params.get("code", [None])[0]

    if auth_code and not st.session_state.get('credentials'):
        st.session_state.auth_flow.fetch_token(code=auth_code)
        st.session_state.credentials = st.session_state.auth_flow.credentials

    if not st.session_state.get('credentials'):
        show_google_sign_in(st.session_state.auth_url)
    else:
        init_session_state()
        account = auth_search_console(client_config, st.session_state.credentials)
        properties = list_gsc_properties(st.session_state.credentials)

        if properties:
            webproperty = show_property_selector(properties, account)
            search_type = show_search_type_selector()
            date_range_selection = show_date_range_selector()
            start_date, end_date = calc_date_range(date_range_selection)
            selected_dimensions = show_dimensions_selector(search_type)
            max_position = show_max_position_selector()
            min_clicks = show_min_clicks_input()
            brand_keywords = st_tags(value=[], suggestions=[], label="Brand Keywords", text="Enter brand keywords to exclude", maxtags=-1, key="brand_keywords")
            show_fetch_data_button(webproperty, search_type, start_date, end_date, selected_dimensions, max_position, min_clicks, brand_keywords)
            if 'fetched_data' in st.session_state and st.session_state.fetched_data is not None:
                for n in range(1, 5):  # For n-grams of length 1 to 4
                    ngrams_df, fig = process_and_plot_ngrams(st.session_state.fetched_data, numGrams=n)
                    st.plotly_chart(fig, use_container_width=True)
                    show_dataframe(ngrams_df)
                    download_csv_link(ngrams_df)
            
if __name__ == "__main__":
    main()
