import pandas as pd
import datetime
import uuid
import altair as alt
import httpagentparser
import os
import requests

def _get_ipinfo(ip):
    url = f"https://ipinfo.io/{ip}/json"
    response = requests.get(url, timeout=5)
    return response.json()

class AnalyticsData:
    """
    An in memory persistence object. Using Pandas DataFrames
    """

    def __init__(self):
        # Define file paths for persistence
        self.data_dir = "myapp/analytics_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        self.sessions_file = os.path.join(self.data_dir, "sessions.csv")
        self.requests_file = os.path.join(self.data_dir, "requests.csv")
        self.clicks_file = os.path.join(self.data_dir, "clicks.csv")

        # --- STAR SCHEMA ---
        
        # 1. SESSION Table (Dimension)
        if os.path.exists(self.sessions_file):
            self.sessions = pd.read_csv(self.sessions_file)
        else:
            self.sessions = pd.DataFrame(columns=[
                'session_id', 'user_id', 'ip_address', 'user_agent', 
                'browser', 'os', 'city', 'country', 'start_time', 'end_time', 'duration'
            ])

        # 2. REQUEST Table (Fact)
        if os.path.exists(self.requests_file):
            self.requests = pd.read_csv(self.requests_file)
        else:
            self.requests = pd.DataFrame(columns=[
                'request_id', 'session_id', 'url', 'method', 
                'query_terms', 'term_count', 'search_method', 'timestamp'
            ])

        # 3. CLICK Table (Fact)
        if os.path.exists(self.clicks_file):
            self.clicks = pd.read_csv(self.clicks_file)
        else:
            self.clicks = pd.DataFrame(columns=[
                'click_id', 'session_id', 'doc_id', 'rank', 'timestamp', 'dwell_time'
            ])



    def save_session(self, session_id: str, user_id: str, ip_address: str, user_agent: str):
        """Logs a new session with user context"""
        # Ensure user_agent is a string
        if user_agent is None:
            user_agent = ""

        data = _get_ipinfo(ip_address)

        now = datetime.datetime.now()

        if session_id not in self.sessions['session_id'].values:
            try:
                agent = httpagentparser.detect(user_agent)
                browser = agent.get('browser', {}).get('name', 'Unknown')
                os_name = agent.get('os', {}).get('name', 'Unknown')
            except Exception as e:
                print(f"Error parsing user agent: {e}")
                browser = 'Unknown'
                os_name = 'Unknown'
            
            new_row = {
                'session_id': session_id,
                'user_id': user_id,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'browser': browser,
                'os': os_name,
                'city': data['city'],
                'country': data['country'],
                'start_time': now,
                'end_time': now,
                'duration': 0.0
            }
            print(f"Saving new session: {session_id}")
            self.sessions = pd.concat([self.sessions, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Update existing session
            mask = self.sessions['session_id'] == session_id
            idx = self.sessions.index[mask][0]
            
            start_time = self.sessions.at[idx, 'start_time']
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            
            duration = (now - start_time).total_seconds()
            
            self.sessions.at[idx, 'end_time'] = now
            self.sessions.at[idx, 'duration'] = duration
            
        self.sessions.to_csv(self.sessions_file, index=False)

    def save_request(self, session_id: str, url: str, method: str = 'GET', query_terms: str = None, search_method: str = None) -> str:
        """Logs an HTTP request. Returns request_id."""
        request_id = str(uuid.uuid4())
        term_count = len(query_terms.split()) if query_terms else 0
        
        new_row = {
            'request_id': request_id,
            'session_id': session_id,
            'url': url,
            'method': method,
            'query_terms': query_terms,
            'term_count': term_count,
            'search_method': search_method,
            'timestamp': datetime.datetime.now()
        }
        self.requests = pd.concat([self.requests, pd.DataFrame([new_row])], ignore_index=True)
        self.requests.to_csv(self.requests_file, index=False)
        return request_id

    def save_click(self, session_id: str, doc_id: str, rank: int):
        """Logs a click event"""
        click_id = str(uuid.uuid4())
        
        new_row = {
            'click_id': click_id,
            'session_id': session_id,
            'doc_id': doc_id,
            'rank': rank,
            'timestamp': datetime.datetime.now(),
            'dwell_time': 0.0
        }
        self.clicks = pd.concat([self.clicks, pd.DataFrame([new_row])], ignore_index=True)
        self.clicks.to_csv(self.clicks_file, index=False)
        return click_id

    def log_dwell_time(self, click_id: str, dwell_time: float):
        """Updates the dwell time for a specific click event"""
        if click_id in self.clicks['click_id'].values:
            mask = self.clicks['click_id'] == click_id
            idx = self.clicks.index[mask][0]
            self.clicks.at[idx, 'dwell_time'] = dwell_time
            self.clicks.to_csv(self.clicks_file, index=False)
            print(f"Logged dwell time {dwell_time}s for click {click_id}")
        else:
            print(f"Click ID {click_id} not found for logging dwell time")



    # --- HELPERS FOR DASHBOARD --- #
    def get_clicks_per_doc(self) -> pd.DataFrame:
        """Helper for the dashboard to count clicks per document"""
        if self.clicks.empty:
            return {}
        return self.clicks['doc_id'].value_counts().to_dict()

    def plot_number_of_views(self):
        if self.clicks.empty:
            return "<div>No clicks recorded yet.</div>"
        
        # Group by doc_id to count views
        df_counts = self.clicks['doc_id'].value_counts().reset_index()
        df_counts.columns = ['Document ID', 'Number of Views']

        chart = alt.Chart(df_counts).mark_bar().encode(
            x='Document ID',
            y='Number of Views',
            tooltip=['Document ID', 'Number of Views']
        ).properties(
            title='Number of Views per Document'
        ).interactive()

        return chart.to_html()

    def get_kpis(self):
        """Returns basic Key Performance Indicators"""
        total_sessions = len(self.sessions)
        # Count requests that have query terms (searches)
        total_searches = len(self.requests[self.requests['query_terms'].notna()]) if not self.requests.empty else 0
        total_clicks = len(self.clicks)
        
        avg_session_duration = 0
        if not self.sessions.empty and 'duration' in self.sessions.columns:
             avg_session_duration = round(self.sessions['duration'].mean() / 60, 2)
             
        avg_dwell_time = 0
        if not self.clicks.empty and 'dwell_time' in self.clicks.columns:
            avg_dwell_time = round(self.clicks['dwell_time'].mean(), 2)

        return {
            'total_sessions': total_sessions,
            'total_searches': total_searches,
            'total_clicks': total_clicks,
            'avg_session_duration': avg_session_duration,
            'avg_dwell_time': avg_dwell_time
        }

    def get_dwell_time_per_doc(self):
        """Returns average dwell time per document"""
        if self.clicks.empty or 'dwell_time' not in self.clicks.columns:
            return {}
        return self.clicks.groupby('doc_id')['dwell_time'].mean().to_dict()

    def get_avg_rank_per_doc(self):
        """Returns average rank per document"""
        if self.clicks.empty or 'rank' not in self.clicks.columns:
            return {}
        return self.clicks.groupby('doc_id')['rank'].mean().to_dict()

    def get_browser_stats(self):
        """Returns browser distribution"""
        if self.sessions.empty: return {}
        return self.sessions['browser'].value_counts().to_dict()

    def get_os_stats(self):
        """Returns OS distribution"""
        if self.sessions.empty: return {}
        return self.sessions['os'].value_counts().to_dict()
    
    def get_country_stats(self):
        """Returns session count per country"""
        if self.sessions.empty:
            return {}
        # Assume 'country' column exists (from ipinfo)
        return self.sessions['country'].value_counts().to_dict()
    
    def get_city_stats(self):
        """Returns session count per country"""
        if self.sessions.empty:
            return {}
        # Assume 'country' column exists (from ipinfo)
        return self.sessions['city'].value_counts().to_dict()
    
    def get_country_city_stats(self):
        if self.sessions.empty:
            return {}
        data = {}
        for _, row in self.sessions.iterrows():
            country = row["country"]
            city = row.get("city", "Unknown")
            data.setdefault(country, {})
            data[country][city] = data[country].get(city, 0) + 1
        return data
    
    def get_top_queries(self, limit=10):
        """Returns top search queries"""
        if self.requests.empty: return {}
        # Filter only search requests (where query_terms is not null/nan)
        searches = self.requests[self.requests['query_terms'].notna()]
        if searches.empty: return {}
        return searches['query_terms'].value_counts().head(limit).to_dict()

    def get_search_methods(self):
        """Returns search method distribution"""
        if self.requests.empty: return {}
        if 'search_method' not in self.requests.columns: return {}
        
        # Filter only search requests (where search_method is not null/nan and not empty)
        # We use str.len() > 0 to ensure we have actual text
        searches = self.requests[
            self.requests['search_method'].notna() & 
            (self.requests['search_method'] != '')
        ]
        
        if searches.empty: return {}
        return searches['search_method'].value_counts().to_dict()

    def get_daily_traffic(self):
        """Returns sessions count per day"""
        if self.sessions.empty: return {}
        # Ensure start_time is datetime
        self.sessions['start_time'] = pd.to_datetime(self.sessions['start_time'])
        daily = self.sessions.groupby(self.sessions['start_time'].dt.date).size()
        # Convert keys to string for JSON serialization
        return {str(k): v for k, v in daily.items()}
    # ----------------------------- #
