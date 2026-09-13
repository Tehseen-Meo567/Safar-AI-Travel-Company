import streamlit as st
from itinerary_utils import generate_itinerary, format_itinerary, create_travel_map

st.set_page_config(page_title="SAFAR - Itinerary & Map", page_icon="🗺️", layout="wide")

st.title("🗺️ SAFAR: AI Itinerary & Travel Map")
st.caption("Generates a day-by-day plan and route map from your budget & hotel picks.")

# In a Streamlit multi-page app, st.session_state persists across pages,
# so this reads whatever the Budget & Hotels page (app_1_.py) saved earlier.
trip_context = st.session_state.get("trip_context")

if trip_context is None:
    st.warning(
        "No trip data found yet. Please complete the **Budget & Hotels** step first — "
        "this page needs your destinations, budget, and hotel picks to build an itinerary."
    )
    st.stop()

col1, col2 = st.columns(2)
with col1:
    tourist_name = st.text_input("👤 Your Name", value="Traveler")
with col2:
    country = st.text_input("🌍 Your Country", placeholder="e.g. United Kingdom")

interests = st.text_input(
    "❤️ Travel Interests",
    placeholder="e.g. Culture, food, history, mountains, shopping...",
)

if st.button("✨ Generate My Safar Plan", type="primary"):
    with st.spinner("Generating your itinerary..."):
        itinerary, error = generate_itinerary(
            trip_context, tourist_name=tourist_name, country=country, interests=interests
        )

    if error:
        st.error(error)
    else:
        # Save so the Chatbot page can reference it too.
        st.session_state["itinerary"] = itinerary

        st.markdown(format_itinerary(itinerary))

        st.subheader("🗺️ Your Visual Travel Route")
        destinations = trip_context.get("destinations", [])
        travel_map = create_travel_map(destinations, trip_context.get("hotel_recommendations"))
        if travel_map is not None:
            st.components.v1.html(travel_map._repr_html_(), height=500)
        else:
            st.info("Map rendering unavailable — check that `folium` is installed.")
