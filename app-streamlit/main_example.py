import streamlit as st
from streamlit_option_menu import option_menu

def on_change(key):
    selection = st.session_state[key]
    st.write(f"Selection changed to {selection}")
    _currentPage = __import__('pages.'+selection+'.index', fromlist=['object']) 
    _currentPage.main()

def streamlit_menu():
    with st.sidebar:
        selected = option_menu(menu_title=None,
            options=["Home", 'Analyze', 'Examples', 'Settings'], 
            icons=['house','database','boxes', 'gear'],
            menu_icon="cast",
            on_change=on_change,
            key='menu',
            default_index=0)
    return selected

# selected = streamlit_menu()

pages = {
    "Your account": [
        st.Page("pages/Analyze/analyze.py", title="Create your account"),
        # st.Page("pages/Analyze/index.py", title="Manage your account"),
    ],
    "Examples": [
         st.Page("pages/Examples/example.py", title="Create your account"),
        # st.Page("pages/Examples/index.py", title="Learn about us"),
        # st.Page("pages/Analyze/index.py", title="Try it out"),
    ],
}

pg = st.navigation(pages)
pg.run()

