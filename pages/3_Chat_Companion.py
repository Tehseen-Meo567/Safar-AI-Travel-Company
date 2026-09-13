import streamlit as st
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. APP SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Chat Companion | Pakistan Travel",
    page_icon="🇵🇰",
    layout="centered",
)

# ---------------------------------------------------------
# 2. TRAVEL KNOWLEDGE BASE
#    This is the RAG source material used by the app.
# ---------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "title": "Visa & entry",
        "text": (
            "For Pakistan tourist visa guidance, direct travelers to the official "
            "NADRA Pakistan Online Visa portal: https://visa.nadra.gov.pk/. "
            "Tourists should carry 10-15 printed physical photocopies of their "
            "passport and e-Visa Grant Notice for possible police checkpoints, "
            "especially when traveling in Gilgit-Baltistan and Khyber Pakhtunkhwa. "
            "No Objection Certificates (NOCs) are generally not needed for the "
            "main tourist hubs of Hunza, Skardu, Swat, Lahore, and Islamabad, "
            "but restricted border zones can require special clearance."
        ),
    },
    {
        "title": "Clothing & cultural etiquette",
        "text": (
            "Recommend loose, modest clothing. Shalwar Kameez is widely appreciated. "
            "For all genders, shoulders and knees should generally be covered. "
            "Women should carry a scarf or dupatta for mosques and conservative areas. "
            "Pakistan has a strong tradition of Mehman Nawazi, meaning hospitality to guests. "
            "A hand-on-heart gesture can be used as a respectful greeting, especially "
            "when physical contact is not appropriate."
        ),
    },
    {
        "title": "Useful Urdu",
        "text": (
            "Useful phrases for visitors include: Assalam-o-Alaikum (hello/greeting), "
            "Shukriya (thank you), and Kitnay ka hai? (how much is this?). "
            "Keep Urdu translations short and provide simple phonetic guidance when useful."
        ),
    },
    {
        "title": "Religious respect",
        "text": (
            "When visiting a mosque, remove your shoes, dress modestly, keep your voice low, "
            "and follow local instructions. A head covering is appropriate for women and can "
            "also be used by men where requested or customary. Be respectful around prayer "
            "times and the Azan. During Ramadan, avoid eating, drinking, or smoking openly "
            "in front of people who are fasting, especially in conservative public settings."
        ),
    },
    {
        "title": "Money & ATMs",
        "text": (
            "Cash in Pakistani Rupees (PKR) is important outside major city malls and other "
            "card-friendly locations. For international card withdrawals, suggest ATMs from "
            "major banks such as HBL, UBL, and Meezan where available. In traditional bazaars, "
            "bargaining can be normal. Bargain politely, keep a friendly tone, and accept the "
            "seller's final price if you decide not to continue."
        ),
    },
    {
        "title": "Getting around cities",
        "text": (
            "In major cities, recommend ride-hailing services such as Careem, Yango, or inDrive "
            "instead of unmetered street taxis. Confirm the destination and fare in the app before "
            "starting the ride. For intercity travel, commonly used options include Daewoo Express, "
            "Faisal Movers, and domestic flights."
        ),
    },
    {
        "title": "Photography & security",
        "text": (
            "Never photograph military installations, check-posts, airports, or government buildings. "
            "When photographing local people, ask permission first. Give balanced, practical advice "
            "rather than sensationalizing security concerns. Follow instructions from police, security "
            "staff, and local authorities."
        ),
    },
    {
        "title": "Emergency contacts",
        "text": (
            "Common emergency contacts for travelers are Police: 15, "
            "Medical Emergency: 1122, and Tourist Helpline: 1422. "
            "In an emergency, contact the appropriate local authority and follow their instructions."
        ),
    },
]


# ---------------------------------------------------------
# 3. RAG RETRIEVAL
#    Beginner-friendly: TF-IDF finds the most relevant
#    knowledge-base passages before asking Groq to answer.
# ---------------------------------------------------------
DOCUMENTS = [item["text"] for item in KNOWLEDGE_BASE]
VECTORIZER = TfidfVectorizer(stop_words="english")
DOCUMENT_MATRIX = VECTORIZER.fit_transform(DOCUMENTS)


def retrieve_context(question, top_k=4):
    """Return the most relevant knowledge-base passages."""
    question_vector = VECTORIZER.transform([question])
    scores = cosine_similarity(question_vector, DOCUMENT_MATRIX)[0]

    ranked_indexes = scores.argsort()[::-1][:top_k]

    results = []
    for index in ranked_indexes:
        if scores[index] > 0:
            results.append(
                {
                    "title": KNOWLEDGE_BASE[index]["title"],
                    "text": KNOWLEDGE_BASE[index]["text"],
                    "score": float(scores[index]),
                }
            )

    return results


# ---------------------------------------------------------
# 4. GROQ
# ---------------------------------------------------------
def get_groq_client():
    """Read the Groq API key from Streamlit Secrets."""
    if "GROQ_API_KEY" not in st.secrets:
        return None
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def generate_answer(question, chat_history, retrieved_chunks):
    """Send the user's question + retrieved RAG context to Groq."""

    client = get_groq_client()
    if client is None:
        return (
            "The Groq API key has not been added yet. "
            "Please add GROQ_API_KEY in Streamlit Community Cloud → "
            "your app → Settings → Secrets."
        )

    context = "\n\n".join(
        f"[{chunk['title']}]\n{chunk['text']}" for chunk in retrieved_chunks
    )

    system_prompt = f"""
You are "Chat Companion", a friendly, respectful, safety-conscious local
Pakistani travel companion for foreign tourists visiting Pakistan.

Your job is to reduce anxiety for first-time visitors with short, practical,
mobile-friendly advice.

IMPORTANT RULES:
1. Use the supplied knowledge base as your primary source.
2. Do not invent visa rules, permits, emergency numbers, security procedures,
   prices, opening hours, transport schedules, or legal requirements.
3. If the knowledge base does not answer something, clearly say that the app
   does not have enough information and recommend checking an official source
   or local authority.
4. For visa applications, direct users only to the official NADRA portal:
   https://visa.nadra.gov.pk/
5. Do not claim that an NOC is never required. Explain that the main tourist
   hubs listed in the knowledge base generally do not require one, while
   restricted border zones can require special clearance.
6. Never encourage photography of military installations, check-posts,
   airports, or government buildings.
7. Ask permission before suggesting photographs of local people.
8. Be culturally respectful and avoid stereotypes.
9. Use headings and short bullet points when helpful.
10. When giving Urdu phrases, include English meaning and a simple phonetic
    pronunciation.
11. If the user asks about an immediate emergency, prioritize emergency
    contacts and following local authorities' instructions.
12. Do not present this chatbot as a replacement for an embassy, immigration
    authority, police officer, doctor, or other official authority.

RETRIEVED KNOWLEDGE:
{context}
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Keep only the most recent messages so prompts stay manageable.
    messages.extend(chat_history[-8:])
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=st.secrets.get("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )
        return response.choices[0].message.content
    except Exception as error:
        return (
            "I couldn't reach the AI service right now. "
            "Please check that your Groq API key is correct and try again.\n\n"
            f"Technical detail: {error}"
        )


# ---------------------------------------------------------
# 5. USER INTERFACE
# ---------------------------------------------------------
st.title("🇵🇰 Chat Companion")
st.caption("A practical AI travel companion for foreign tourists visiting Pakistan.")

with st.sidebar:
    st.header("About")
    st.write(
        "This beginner-friendly RAG app retrieves relevant Pakistan travel "
        "guidance first, then asks Groq to turn that information into a helpful answer."
    )

    st.markdown("### Quick topics")
    st.write("• Visa & entry")
    st.write("• Culture & clothing")
    st.write("• Urdu phrases")
    st.write("• Money & ATMs")
    st.write("• Transport")
    st.write("• Photography & security")
    st.write("• Emergency contacts")

    st.markdown("### Official visa portal")
    st.link_button(
        "Open NADRA e-Visa",
        "https://visa.nadra.gov.pk/",
        use_container_width=True,
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []


# Welcome message for a first-time visitor.
if not st.session_state.messages:
    st.info(
        "👋 Welcome! Ask me something like: "
        "“What should I wear in Lahore?” or "
        "“What should I carry for a trip to Hunza?”"
    )


# Show previous messages.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# 6. NEW QUESTION
# ---------------------------------------------------------
question = st.chat_input("Ask your Pakistan travel question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    retrieved_chunks = retrieve_context(question)

    with st.chat_message("assistant"):
        with st.spinner("Checking the travel knowledge base..."):
            answer = generate_answer(
                question=question,
                chat_history=st.session_state.messages[:-1],
                retrieved_chunks=retrieved_chunks,
            )

        st.markdown(answer)

        # Show the retrieved RAG sources for transparency.
        if retrieved_chunks:
            with st.expander("🔎 Knowledge used for this answer"):
                for chunk in retrieved_chunks:
                    st.markdown(f"**{chunk['title']}**")
                    st.caption(f"Relevance score: {chunk['score']:.2f}")
                    st.write(chunk["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

st.divider()
st.caption(
    "Travel information can change. For visa, border, security, and emergency "
    "matters, verify with the relevant official authority."
)
