import os
import time
import pandas as pd
from json import JSONEncoder
import uuid

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData #, ClickedDoc
from myapp.search.load_corpus import our_load_corpus
#from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
#corpus = load_corpus(file_path)
og_corpus, corpus = our_load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
#print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])
print("\nCorpus is loaded... \n First element:\n", corpus.head(1))


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['GET'])
def search():
    search_query = request.args.get('search-query')
    search_method = request.args.get('search-method', 'tfidf') # Default to tfidf

    # --- DATA ANALYTICS STEPS --- #
    # 1. Capture context info: IP address and user agent
    user_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent')

    # 2. Store last search query in session
    # Updated to pass session_id
    # We now log this as a REQUEST with query terms
    search_id = analytics_data.save_request(
        session_id=session['session_id'], 
        url=request.path, 
        method=request.method, 
        query_terms=search_query,
        search_method=search_method
    )

    # 3. Save search_id to session
    session['last_search_id'] = search_id
    session['last_search_query'] = search_query
    # --- End of DATA ANALYTICS STEPS --- #

    start_time = time.time()

    # -- SEARCH FUNCTION -- #
    results = search_engine.search(search_query, search_id, og_corpus, corpus, search_method)

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count
    print(session)

    elapsed_time = time.time() - start_time

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count, time_taken=elapsed_time, rag_response=rag_response)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    # --- DATA ANALYTICS STEPS --- #

    # get the query string parameters from request
    clicked_doc_id = request.args["pid"]
    # Get rank if available (passed from results.html loop index)
    rank = request.args.get("rank", 1) 
    try:
        rank = int(rank)
    except:
        rank = 1

    print("click in id={}".format(clicked_doc_id))

    # 1. Recover search_id from session
    # In our new schema, the search_id is actually the request_id of the search request
    search_id = session.get('last_search_id', None)
    
    # 2. Log the click into analytics_data
    # We link the click to the session, not just the search_id (though we could link both if we added a column)
    # The requirement says "to what query where related", so we can use search_id if we want, 
    # but the schema requested was Session, Click, Request. 
    # Let's assume Click links to Session, and we can infer the query from the Session's recent requests.
    # However, to be precise, let's pass the session_id.
    click_id = analytics_data.save_click(session['session_id'], clicked_doc_id, rank)

    # --- End of DATA ANALYTICS STEPS --- #


    # --- OUR CODE TO RETRIEVE DOCUMENT DETAILS --- #
    # Retrieve the document from the original corpus
    document = None
    matching_rows = og_corpus.loc[og_corpus['pid'] == clicked_doc_id]
    if not matching_rows.empty:
        document = matching_rows.iloc[0].to_dict()
        
        # Clean up NaNs for better template rendering
        for key, value in document.items():
            if isinstance(value, (list, dict)):
                continue
            if pd.isna(value):
                document[key] = None

    return render_template('doc_details.html', document=document, click_id=click_id)


@app.route('/dashboard', methods=['GET'])
def dashboard():
    # Get all the data needed for the dashboard
    kpis = analytics_data.get_kpis()
    browser_stats = analytics_data.get_browser_stats()
    os_stats = analytics_data.get_os_stats()
    top_queries = analytics_data.get_top_queries()
    search_methods = analytics_data.get_search_methods()
    daily_traffic = analytics_data.get_daily_traffic()
    
    # Existing logic for visited docs
    clicks_dict = analytics_data.get_clicks_per_doc()
    dwell_time_dict = analytics_data.get_dwell_time_per_doc()
    avg_rank_dict = analytics_data.get_avg_rank_per_doc()
    
    visited_docs = []
    for doc_id, count in clicks_dict.items():
        # Find the document in the corpus
        matching_rows = og_corpus.loc[og_corpus['pid'] == doc_id]
        if not matching_rows.empty:
            row = matching_rows.iloc[0]
            doc = {
                'doc_id': doc_id,
                'title': row['title'],
                'counter': count,
                'avg_dwell_time': round(dwell_time_dict.get(doc_id, 0), 2),
                'avg_rank': round(avg_rank_dict.get(doc_id, 0), 1)
            }
            visited_docs.append(doc)

    visited_docs.sort(key=lambda doc: doc['counter'], reverse=True)
    
    return render_template('dashboard.html', 
                           visited_docs=visited_docs,
                           kpis=kpis,
                           browser_stats=browser_stats,
                           os_stats=os_stats,
                           top_queries=top_queries,
                           search_methods=search_methods,
                           daily_traffic=daily_traffic)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()
@app.route('/log_dwell_time', methods=['POST'])
def log_dwell_time():
    data = request.get_json()
    click_id = data.get('click_id')
    dwell_time = data.get('dwell_time')
    
    if click_id and dwell_time is not None:
        analytics_data.log_dwell_time(click_id, float(dwell_time))
        return "OK", 200
    return "Bad Request", 400


@app.before_request
def before_request():
    # 1. Ensure User ID (Visitor Context)
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    
    # 2. Ensure Session ID (Physical Session)
    # Logic: New session if none exists or last activity > 30 mins ago
    now = time.time()
    last_activity = session.get('last_activity', 0)
    if 'session_id' not in session or (now - last_activity > 1800):
        session['session_id'] = str(uuid.uuid4())
        
    # Ensure session is logged in analytics (idempotent check inside save_session)
    user_agent = request.headers.get('User-Agent')
    user_ip = request.remote_addr
    analytics_data.save_session(session['session_id'], session['user_id'], user_ip, user_agent)
    
    session['last_activity'] = now

    # 3. Log User Context (if new user/session)
    # We can do this once per session to be safe
    # user_agent = request.headers.get('User-Agent')
    # user_ip = request.remote_addr
    # analytics_data.save_user(session['user_id'], user_ip, user_agent)

    # 4. Log Page View (HTTP Requests data)
    # Only log GET requests here to avoid double logging POST searches
    # Also exclude /search because it is logged separately with query terms
    if request.method == 'GET' and request.endpoint != 'static' and request.path != '/search':
         analytics_data.save_request(session['session_id'], request.path, request.method)


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))
