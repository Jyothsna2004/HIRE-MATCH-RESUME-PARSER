import streamlit as st
st.set_page_config(page_title="Enhanced Resume Role Suitability", layout="wide")

import pickle
import PyPDF2
import docx
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please install spaCy English model: python -m spacy download en_core_web_sm")
    nlp = None

# Load trained model and components
try:
    clf = pickle.load(open("rf_clf.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    le = pickle.load(open("encoder.pkl", "rb"))
    embedder = pickle.load(open("bert_embedder.pkl", "rb"))
except:
    st.error("Model files not found. Please ensure rf_clf.pkl, tfidf.pkl, encoder.pkl, and bert_embedder.pkl are in the same directory.")
    st.stop()

# Load sentence transformer for semantic similarity
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = load_sentence_transformer()

# Clean text - improved version
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', " ", text)
    
    # Remove phone numbers
    text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', " ", text)
    
    # Keep alphanumeric characters and common punctuation
    text = re.sub(r"[^a-zA-Z0-9\s\-\+\#\.\,]", " ", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip().lower()

# Advanced text preprocessing
def preprocess_text_advanced(text):
    # Remove URLs, emails, phone numbers
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r'\S+@\S+', " ", text)
    text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', " ", text)
    
    # Keep alphanumeric and important symbols
    text = re.sub(r"[^a-zA-Z0-9\s\-\+\#\.\,]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize and remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        # Add common resume stopwords
        stop_words.update(['college', 'university', 'school', 'institute', 'academy', 'ltd', 'inc', 'corp'])
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words and len(word) > 2]
        return ' '.join(filtered_text)
    except:
        return text

# Improved skills and certifications extraction
def extract_skills_and_certifications(text):
    skills = set()
    certifications = set()
    
    # Define comprehensive skill patterns
    programming_languages = [
        'python', 'java', 'javascript', 'c', 'c++', 'php', 'ruby', 'go', 'rust', 'swift',
        'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'typescript'
    ]
    
    frameworks_libraries = [
        'react', 'angular', 'vue', 'django', 'flask', 'spring', 'nodejs', 'express',
        'bootstrap', 'jquery', 'pandas', 'numpy', 'matplotlib', 'plotly', 'tensorflow',
        'pytorch', 'scikit-learn', 'opencv', 'selenium', 'junit'
    ]
    
    databases = [
        'mongodb', 'mysql', 'postgresql', 'sqlite', 'redis', 'cassandra', 'oracle',
        'sql server', 'dynamodb', 'firebase'
    ]
    
    cloud_tools = [
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
        'gitlab', 'bitbucket', 'terraform', 'ansible', 'linux', 'ubuntu', 'windows'
    ]
    
    # Combine all technical skills
    all_tech_skills = programming_languages + frameworks_libraries + databases + cloud_tools
    
    # Extract skills from text
    text_lower = text.lower()
    
    # Direct skill matching
    for skill in all_tech_skills:
        if skill in text_lower:
            skills.add(skill)
    
    # Pattern-based extraction for certifications
    cert_patterns = [
        r'gold medal in ([a-zA-Z\s]+)',
        r'certification in ([a-zA-Z\s]+)',
        r'certified in ([a-zA-Z\s]+)',
        r'completed ([a-zA-Z\s]+) course',
        r'([a-zA-Z\s]+) certification',
        r'([a-zA-Z\s]+) certified',
        r'diploma in ([a-zA-Z\s]+)',
        r'degree in ([a-zA-Z\s]+)',
        r'specialization in ([a-zA-Z\s]+)',
        r'training in ([a-zA-Z\s]+)',
        r'summer school',
        r'ml summer school',
        r'amazon ml',
        r'nptel',
        r'cisco',
        r'ccna'
    ]
    
    for pattern in cert_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, str):
                cert = match.strip()
                if len(cert) > 2 and cert not in ['and', 'the', 'for', 'with', 'from']:
                    certifications.add(cert)
    
    # Extract project-based skills
    project_patterns = [
        r'built using ([a-zA-Z\s,]+)',
        r'developed using ([a-zA-Z\s,]+)',
        r'implemented using ([a-zA-Z\s,]+)',
        r'technologies used: ([a-zA-Z\s,]+)',
        r'stack: ([a-zA-Z\s,]+)',
        r'mern stack',
        r'full.stack',
        r'machine learning',
        r'deep learning',
        r'natural language processing',
        r'nlp',
        r'data science',
        r'web development',
        r'mobile development'
    ]
    
    for pattern in project_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, str):
                # Split by comma and extract individual skills
                skill_list = [s.strip() for s in match.split(',')]
                for skill in skill_list:
                    if len(skill) > 2 and skill not in ['and', 'the', 'for', 'with', 'from']:
                        skills.add(skill)
    
    # Use spaCy NER for additional extraction
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT'] and len(ent.text) > 2:
                entity_text = ent.text.lower()
                # Filter out common non-skill entities
                if entity_text not in ['college', 'university', 'school', 'institute', 'academy']:
                    skills.add(entity_text)
    
    return skills, certifications

# Enhanced keyword matching with better semantic similarity
def enhanced_keyword_matching(resume_text, required_keywords, threshold=0.6):
    """
    Enhanced keyword matching using semantic similarity with lower threshold
    """
    found_keywords = []
    missing_keywords = []
    
    # Extract skills and certifications from resume
    extracted_skills, extracted_certs = extract_skills_and_certifications(resume_text)
    
    # Combine all extracted content
    all_extracted = list(extracted_skills) + list(extracted_certs)
    
    # Get sentences from resume for context
    sentences = sent_tokenize(resume_text)
    
    # Create skill synonyms mapping
    skill_synonyms = {
        'python': ['python programming', 'python development', 'python scripting'],
        'java': ['java programming', 'java development', 'core java'],
        'javascript': ['js', 'node.js', 'nodejs', 'react', 'angular', 'vue'],
        'machine learning': ['ml', 'artificial intelligence', 'ai', 'data science'],
        'web development': ['html', 'css', 'javascript', 'react', 'frontend', 'backend'],
        'database': ['sql', 'mongodb', 'mysql', 'postgresql', 'dbms'],
        'rest apis': ['api', 'rest', 'restful', 'web services'],
        'git': ['github', 'gitlab', 'version control'],
        'aws': ['amazon web services', 'cloud computing', 'cloud'],
        'docker': ['containerization', 'containers'],
        'programming': ['coding', 'development', 'software development']
    }
    
    for required_kw in required_keywords:
        keyword_found = False
        required_kw_lower = required_kw.lower()
        
        # Direct keyword matching (exact or substring)
        if required_kw_lower in resume_text.lower():
            found_keywords.append(required_kw)
            keyword_found = True
            continue
        
        # Check synonyms
        for synonym_key, synonyms in skill_synonyms.items():
            if required_kw_lower == synonym_key or required_kw_lower in synonyms:
                for synonym in synonyms + [synonym_key]:
                    if synonym in resume_text.lower():
                        found_keywords.append(required_kw)
                        keyword_found = True
                        break
                if keyword_found:
                    break
        
        if keyword_found:
            continue
        
        # Check extracted skills and certifications
        for extracted_item in all_extracted:
            if (required_kw_lower in extracted_item or 
                extracted_item in required_kw_lower or
                any(synonym in extracted_item for synonym in skill_synonyms.get(required_kw_lower, []))):
                found_keywords.append(required_kw)
                keyword_found = True
                break
        
        if keyword_found:
            continue
        
        # Semantic similarity matching with lower threshold
        try:
            required_kw_embedding = sentence_model.encode([required_kw])
            
            # Check against extracted skills/certifications
            for extracted_item in all_extracted:
                if len(extracted_item) > 2:
                    extracted_embedding = sentence_model.encode([extracted_item])
                    similarity = cosine_similarity(required_kw_embedding, extracted_embedding)[0][0]
                    
                    if similarity > threshold:
                        found_keywords.append(required_kw)
                        keyword_found = True
                        break
            
            if keyword_found:
                continue
            
            # Check against resume sentences for context
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    sentence_embedding = sentence_model.encode([sentence])
                    similarity = cosine_similarity(required_kw_embedding, sentence_embedding)[0][0]
                    
                    if similarity > threshold:
                        found_keywords.append(required_kw)
                        keyword_found = True
                        break
        except:
            pass
        
        if not keyword_found:
            missing_keywords.append(required_kw)
    
    return found_keywords, missing_keywords

# Enhanced skill synonyms and related terms
def get_enhanced_skill_synonyms():
    return {
        "python": ["python programming", "python development", "python scripting", "python course", "python certification", "python training"],
        "java": ["java programming", "java development", "core java", "java course", "java certification", "java training"],
        "machine learning": ["ml", "artificial intelligence", "ai", "data science", "predictive modeling", "deep learning"],
        "database": ["sql", "mysql", "postgresql", "mongodb", "database management", "dbms"],
        "web development": ["html", "css", "javascript", "react", "angular", "vue", "frontend", "backend"],
        "project management": ["pmp", "agile", "scrum", "project coordination", "team leadership"],
        "data analysis": ["analytics", "data visualization", "statistical analysis", "business intelligence", "excel"],
        "cloud computing": ["aws", "azure", "google cloud", "cloud services", "cloud architecture"],
        "testing": ["quality assurance", "qa", "test automation", "manual testing", "selenium", "junit"]
    }

# Enhanced prediction function
def predict_suitability_enhanced(text, selected_role):
    cleaned_text = clean_text(text)
    
    try:
        embedding = embedder.encode([cleaned_text])
        tfidf_vec = tfidf.transform([cleaned_text]).toarray()
        features = np.hstack((embedding, tfidf_vec))

        probabilities = clf.predict_proba(features)[0]
        role_index = le.transform([selected_role])[0]
        confidence = probabilities[role_index] * 100
        
        # Apply boosting for IT roles based on technical content
        if selected_role == "INFORMATION-TECHNOLOGY":
            # Boost score based on technical skills found
            tech_keywords = ['python', 'java', 'javascript', 'react', 'mongodb', 'machine learning', 
                           'web development', 'programming', 'software', 'development', 'api', 'database']
            
            text_lower = text.lower()
            tech_matches = sum(1 for keyword in tech_keywords if keyword in text_lower)
            
            # Apply boost based on technical content
            if tech_matches >= 5:
                confidence = min(confidence * 2.5, 95)  # Significant boost but cap at 95%
            elif tech_matches >= 3:
                confidence = min(confidence * 1.8, 85)  # Moderate boost
            elif tech_matches >= 1:
                confidence = min(confidence * 1.3, 75)  # Small boost
        
        status = "✅ Suitable" if confidence >= 50 else "❌ Not Suitable"

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_roles = [(le.classes_[i], probabilities[i] * 100) for i in top_indices]
        
        # If IT role confidence was boosted, update the top roles
        if selected_role == "INFORMATION-TECHNOLOGY" and confidence > probabilities[role_index] * 100:
            top_roles = [(selected_role, confidence)] + [role for role in top_roles if role[0] != selected_role][:2]
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        confidence = 0
        status = "❌ Error in prediction"
        top_roles = []

    # Enhanced keyword requirements with more comprehensive lists
    keyword_suggestions = {
        "INFORMATION-TECHNOLOGY": [
            "Python", "Java", "JavaScript", "React", "Node.js", "MongoDB", "SQL", 
            "Git", "REST APIs", "Web Development", "Machine Learning", "Programming", 
            "Software Development", "Database", "HTML", "CSS", "Framework"
        ],
        "HR": ["recruitment", "payroll", "training", "employee engagement", "compliance", "HRMS", "onboarding", "human resources"],
        "SALES": ["lead generation", "CRM", "sales target", "client communication", "negotiation", "pipeline", "sales experience"],
        "DESIGNER": ["Figma", "Sketch", "Adobe XD", "Illustrator", "Photoshop", "UI/UX", "prototyping", "design"],
        "FINANCE": ["financial reporting", "budgeting", "accounts", "ledger", "forecasting", "compliance", "audit", "finance"],
        "DATA-SCIENCE": ["Python", "Pandas", "NumPy", "machine learning", "data analysis", "statistics", "Jupyter", "scikit-learn", "data science"],
        "JAVA-DEVELOPER": ["Java", "Spring Boot", "REST APIs", "Maven", "JPA", "Hibernate", "microservices", "java programming"],
        "PYTHON-DEVELOPER": ["Python", "Flask", "Django", "NumPy", "Pandas", "APIs", "web scraping", "python programming"],
        "WEB-DEVELOPER": ["HTML", "CSS", "JavaScript", "React", "Node.js", "MongoDB", "Bootstrap", "Git", "web development"],
        "TESTING": ["manual testing", "test cases", "bug tracking", "Selenium", "JUnit", "automation", "regression", "quality assurance"],
        "NETWORKING": ["TCP/IP", "DNS", "firewall", "routing", "switches", "network protocols", "Wireshark", "networking"],
        "DEVOPS": ["CI/CD", "Docker", "Kubernetes", "Jenkins", "Git", "cloud", "monitoring", "Terraform", "devops"],
        "ANDROID-DEVELOPER": ["Android", "Java", "Kotlin", "XML", "Android Studio", "Firebase", "SQLite", "mobile development"],
        "BLOCKCHAIN": ["smart contracts", "Solidity", "Ethereum", "web3", "DeFi", "Metamask", "ganache", "blockchain"],
        "MECHANICAL-ENGINEER": ["AutoCAD", "SolidWorks", "mechanical design", "CAD", "CAM", "manufacturing", "engineering"],
        "ELECTRICAL-ENGINEER": ["circuits", "PCB", "MATLAB", "Simulink", "multimeter", "schematics", "electrical engineering"],
        "CIVIL-ENGINEER": ["AutoCAD", "site supervision", "construction", "estimating", "BIM", "project management", "civil engineering"],
        "TEACHER": ["lesson planning", "curriculum", "classroom management", "teaching", "assessment", "pedagogy", "education"],
        "CONTENT-WRITER": ["SEO", "blogging", "copywriting", "proofreading", "plagiarism-free", "editing", "CMS", "content writing"],
        "MARKETING": ["SEO", "Google Ads", "campaign", "branding", "analytics", "social media", "email marketing", "marketing"],
        "CUSTOMER-SERVICE": ["CRM", "ticketing", "client interaction", "query resolution", "follow-up", "support", "customer service"],
        "ENGINEERING": ["design", "research", "CAD", "simulation", "materials", "manufacturing", "engineering"],
        "ARTS": ["illustration", "drawing", "creative writing", "literature", "design thinking", "arts", "creative"]
    }

    required_keywords = keyword_suggestions.get(selected_role.upper(), [])
    
    # Use enhanced keyword matching
    found_keywords, missing_keywords = enhanced_keyword_matching(text, required_keywords)
    
    # Extract skills and certifications for display
    extracted_skills, extracted_certs = extract_skills_and_certifications(text)
    
    # Filter out irrelevant extractions
    filtered_skills = {skill for skill in extracted_skills if len(skill) > 2 and 
                      skill not in ['college', 'university', 'school', 'institute', 'academy', 'ltd', 'inc']}
    
    filtered_certs = {cert for cert in extracted_certs if len(cert) > 2 and 
                     cert not in ['and', 'the', 'for', 'with', 'from']}

    learning_resources = {
        "INFORMATION-TECHNOLOGY": [
            "📘 Learn Python from Codecademy or Coursera",
            "🧠 Build small projects like a ToDo app using Flask/Django",
            "☁ Explore AWS Cloud Practitioner Essentials",
            "🔧 Learn Docker basics from YouTube tutorials"
        ],
        "ENGINEERING": [
            "⚙ Brush up on core engineering concepts",
            "💻 Try coding challenges on HackerRank",
            "📊 Learn data visualization using Matplotlib/Seaborn"
        ],
        "ARTS": [
            "🎨 Build a personal portfolio website",
            "📝 Take content writing or graphic design courses on Udemy",
            "📷 Explore Canva or Adobe Express for design"
        ]
    }

    return status, confidence, top_roles, found_keywords, missing_keywords, filtered_skills, filtered_certs, learning_resources.get(selected_role.upper(), [])

# PDF extraction
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# DOCX extraction
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

# TXT extraction
def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except UnicodeDecodeError:
        try:
            file.seek(0)
            return file.read().decode("latin-1")
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""

# Resume text extractor
def get_resume_text(file):
    if file is None:
        return ""
    
    ext = file.name.split('.')[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file)
    elif ext == "docx":
        return extract_text_from_docx(file)
    elif ext == "txt":
        return extract_text_from_txt(file)
    else:
        st.error("Unsupported file format. Please upload PDF, DOCX, or TXT files.")
        return ""

# Enhanced Streamlit UI
def main():
    st.title("📄 HireMatch AI - Enhanced Resume Screening Tool")
    st.markdown("Upload a resume *or* paste text below and select a job role to evaluate how well it fits with *semantic understanding*.")
    
    st.sidebar.markdown("## 🔍 Features")
    st.sidebar.markdown("- *Semantic Understanding*: Understands context and meaning")
    st.sidebar.markdown("- *Skill Extraction*: Automatically extracts skills and certifications")
    st.sidebar.markdown("- *Flexible Matching*: Matches similar concepts (e.g., 'gold medal in python' = 'python certification')")
    st.sidebar.markdown("- *Context Awareness*: Considers sentence context for better matching")
    st.sidebar.markdown("- *Technical Boost*: Special scoring for technical roles")

    selected_role = st.selectbox("🎯 Select Target Role", sorted(le.classes_))

    uploaded_file = st.file_uploader("📎 Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
    st.markdown("### ✏ Or paste resume text directly:")
    typed_text = st.text_area("Paste your resume here", "", height=250)

    resume_text = ""

    if uploaded_file:
        try:
            resume_text = get_resume_text(uploaded_file)
            if resume_text:
                st.success("✅ Resume text extracted successfully from file!")
            else:
                st.error("❌ Could not extract text from file. Please try a different format.")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    elif typed_text.strip():
        resume_text = typed_text
        st.success("✅ Resume text taken from input box!")

    if resume_text and len(resume_text.strip()) > 10:
        if st.checkbox("📄 Show resume text used"):
            st.text_area("Resume Text", resume_text, height=300)

        st.subheader("🔍 Enhanced Suitability Analysis:")
        
        with st.spinner("Analyzing resume with semantic understanding..."):
            try:
                result = predict_suitability_enhanced(resume_text, selected_role)
                status, confidence, top_roles, found_keywords, missing_keywords, extracted_skills, extracted_certs, learning_tips = result
            except Exception as e:
                st.error(f"Error in analysis: {str(e)}")
                return

        # Display results in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Suitability Score")
            
            # Color-coded confidence display
            if confidence >= 70:
                confidence_color = "🟢"
            elif confidence >= 50:
                confidence_color = "🟡"
            else:
                confidence_color = "🔴"
            
            st.metric(label=f"Match for {selected_role}", 
                     value=f"{confidence:.1f}%", 
                     delta=f"{confidence_color} {status}")
            
            st.markdown("### 📌 Top Role Predictions:")
            for i, (role, prob) in enumerate(top_roles[:3]):
                if i == 0:
                    st.markdown(f"🥇 *{role}* — {prob:.1f}%")
                elif i == 1:
                    st.markdown(f"🥈 *{role}* — {prob:.1f}%")
                else:
                    st.markdown(f"🥉 *{role}* — {prob:.1f}%")
        
        with col2:
            st.markdown("### 🎯 Extracted Skills & Certifications:")
            if extracted_skills:
                st.markdown("🔧 Skills Found:")
                skills_list = list(extracted_skills)[:15]  # Show first 15
                for skill in skills_list:
                    st.markdown(f"• {skill}")
            
            if extracted_certs:
                st.markdown("📜 Certifications Found:")
                certs_list = list(extracted_certs)[:8]  # Show first 8
                for cert in certs_list:
                    st.markdown(f"• {cert}")

        # Keyword analysis
        st.markdown("### 🔍 Keyword Analysis:")
        
        col3, col4 = st.columns(2)
        
        with col3:
            if found_keywords:
                st.markdown("#### ✅ *Matching Keywords Found:*")
                for kw in found_keywords:
                    st.markdown(f"• {kw}")
            else:
                st.markdown("#### ❌ *No matching keywords found*")
        
        with col4:
            if missing_keywords:
                st.markdown("#### ❗ *Suggested Keywords to Add:*")
                for kw in missing_keywords:
                    st.markdown(f"• {kw}")
            else:
                st.markdown("#### ✅ *All expected keywords found!*")

        # Show match percentage
        if found_keywords or missing_keywords:
            match_percentage = (len(found_keywords) / (len(found_keywords) + len(missing_keywords))) * 100
            st.markdown(f"*Keyword Match Rate: {match_percentage:.1f}%*")

        if learning_tips:
            st.markdown("### 📚 Suggested Learning Resources:")
            for tip in learning_tips:
                st.markdown(f"• {tip}")

        st.markdown("### 💡 Enhanced Resume Tips:")
        st.markdown("• *Be specific about achievements*: Instead of 'learned Python', write 'achieved gold medal in Python programming course from NPTEL'")
        st.markdown("• *Use action verbs*: 'Implemented', 'Developed', 'Designed', 'Optimized', etc.")
        st.markdown("• *Include metrics*: 'Reduced processing time by 30%', 'Managed team of 5 developers'")
        st.markdown("• *Add certifications and courses*: Any formal training, online courses, or certifications")
        st.markdown("• *Mention tools and technologies*: Be specific about the tools you've used")
        st.markdown("• *Highlight projects*: Include technical details about what you built and how")

    elif resume_text and len(resume_text.strip()) <= 10:
        st.warning("⚠ Resume text is too short. Please provide a more detailed resume.")
    else:
        st.info("📝 Please upload a resume file or paste your resume text to begin analysis.")

if __name__== "__main__":
    main()