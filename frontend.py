import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import os
import time
from datetime import datetime

if 'student_database' not in st.session_state:
    st.session_state.student_database = [
        {"name": "Aayush Chhetri", "roll_number": "THA078BEI001", "email": "aayush.078bei001@tcioe.edu.np", "phone": "9869088844"},
        {"name": "Abhishek Prasad Sah", "roll_number": "THA078BEI002", "email": "abhishek.078bei002@tcioe.edu.np", "phone": "9864106036"},
        {"name": "Agrima Regmi", "roll_number": "THA078BEI003", "email": "agrima.078bei003@tcioe.edu.np", "phone": "9808381522"},
        {"name": "Amrit Kandel", "roll_number": "THA078BEI004", "email": "amrit.078bei004@tcioe.edu.np", "phone": "9866115699"},
        {"name": "Anveshan Timsina", "roll_number": "THA078BEI005", "email": "anveshan.078bei005@tcioe.edu.np", "phone": "9862897754"},
        {"name": "Ashish Kandel", "roll_number": "THA078BEI006", "email": "ashish.078bei006@tcioe.edu.np", "phone": "9842519361"},
        {"name": "Binita Adhikari", "roll_number": "THA078BEI008", "email": "binita.078bei008@tcioe.edu.np", "phone": "9761774565"},
        {"name": "Bipul Kumar Dahal", "roll_number": "THA078BEI009", "email": "bipul.078bei009@tcioe.edu.np", "phone": "9868369363"},
        {"name": "Dinanath Padhya", "roll_number": "THA078BEI010", "email": "dinanath.078bei010@tcioe.edu.np", "phone": "9742893416"},
        {"name": "Dipesh Baral", "roll_number": "THA078BEI011", "email": "dipesh.078bei011@tcioe.edu.np", "phone": "9767487279"},
        {"name": "Dipesh Kadal", "roll_number": "THA078BEI013", "email": "dipesh.078bei013@tcioe.edu.np", "phone": "9828927514"},
        {"name": "Dishan Shakya", "roll_number": "THA078BEI014", "email": "dushan.078bei014@tcioe.edu.np", "phone": "9761797002"},
        {"name": "Diwas Dahal", "roll_number": "THA078BEI015", "email": "diwas.078bei015@tcioe.edu.np", "phone": "9864160968"},
        {"name": "Jatin Raut", "roll_number": "THA078BEI017", "email": "jatin.078bei017@tcioe.edu.np", "phone": "9843041090"},
        {"name": "Jenish Pant", "roll_number": "THA078BEI018", "email": "jenish.078bei018@tcioe.edu.np", "phone": "9841551131"},
        {"name": "Kiman Adhikari", "roll_number": "THA078BEI019", "email": "kiman.078bei019@tcioe.edu.np", "phone": "9843969170"},
        {"name": "Krishna Acharya", "roll_number": "THA078BEI020", "email": "krishna.078bei020@tcioe.edu.np", "phone": "9848046988"},
        {"name": "Mukesh Bhatta", "roll_number": "THA078BEI022", "email": "mukesh.078bei022@tcioe.edu.np", "phone": "9861880765"},
        {"name": "Nabin Shrestha", "roll_number": "THA078BEI023", "email": "nabin.078bei023@tcioe.edu.np", "phone": "9841623316"},
        {"name": "Nischal Bhusal", "roll_number": "THA078BEI024", "email": "nischal.078bei024@tcioe.edu.np", "phone": "9841617360"},
        {"name": "Pankaj Bhatt", "roll_number": "THA078BEI025", "email": "pankaj.078bei025@tcioe.edu.np", "phone": "9865920096"},
        {"name": "Prasish Timalsina", "roll_number": "THA078BEI026", "email": "prasish.078bei026@tcioe.edu.np", "phone": "9745355160"},
        {"name": "Pratik Pokharel", "roll_number": "THA078BEI027", "email": "pratik.078bei027@tcioe.edu.np", "phone": "9867404111"},
        {"name": "Pratistha Sapkota", "roll_number": "THA078BEI028", "email": "pratistha.078bei028@tcioe.edu.np", "phone": "9744341824"},
        {"name": "Pujan Pandey", "roll_number": "THA078BEI029", "email": "pujan.078bei029@tcioe.edu.np", "phone": "9844090278"},
        {"name": "Rabin Rai", "roll_number": "THA078BEI030", "email": "rabin.078bei030@tcioe.edu.np", "phone": "9845350482"},
        {"name": "Roshan Shrestha", "roll_number": "THA078BEI031", "email": "roshan.078bei031@tcioe.edu.np", "phone": "9841253880"},
        {"name": "Roshan Singh Saud", "roll_number": "THA078BEI032", "email": "roshan.078bei032@tcioe.edu.np", "phone": "9840088095"},
        {"name": "Sagar Joshi", "roll_number": "THA078BEI033", "email": "sagar.078bei033@tcioe.edu.np", "phone": "9810079516"},
        {"name": "Sandesh Dhital", "roll_number": "THA078BEI034", "email": "sandesh.078bei034@tcioe.edu.np", "phone": "9849781518"},
        {"name": "Sandesh Panthi", "roll_number": "THA078BEI035", "email": "sandesh.078bei035@tcioe.edu.np", "phone": "9841817893"},
        {"name": "Sanjeep Kumar Sharma", "roll_number": "THA078BEI036", "email": "sanjeep.078bei036@tcioe.edu.np", "phone": "9847379577"},
        {"name": "Sanskriti Khatiwada", "roll_number": "THA078BEI037", "email": "sanskriti.078bei037@tcioe.edu.np", "phone": "9742456719"},
        {"name": "Saroj Nagarkoti", "roll_number": "THA078BEI039", "email": "saroj.078bei039@tcioe.edu.np", "phone": "9841745662"},
        {"name": "Satish Khanal", "roll_number": "THA078BEI040", "email": "satish.078bei040@tcioe.edu.np", "phone": "9869363082"},
        {"name": "Shishir Gaire", "roll_number": "THA078BEI041", "email": "shishir.078bei041@tcioe.edu.np", "phone": "9765556415"},
        {"name": "Subham Gautam", "roll_number": "THA078BEI042", "email": "subham.078bei042@tcioe.edu.np", "phone": "9843787118"},
        {"name": "Subrat Dhital", "roll_number": "THA078BEI043", "email": "subrat.078bei043@tcioe.edu.np", "phone": "9864164424"},
        {"name": "Sudip Kumar Thakur", "roll_number": "THA078BEI044", "email": "sudip.078bei044@tcioe.edu.np", "phone": "9865212075"},
        {"name": "Sujan Gupta", "roll_number": "THA078BEI045", "email": "sujan.078bei045@tcioe.edu.np", "phone": "9818233171"},
        {"name": "Utsab Dahal", "roll_number": "THA078BEI046", "email": "utsab.078bei046@tcioe.edu.np", "phone": "9762227144"},
        {"name": "Matina Tuladhar", "roll_number": "THA078BEI047", "email": "matina.078bei047@tcioe.edu.np", "phone": "9810197047"},
        {"name": "Sajen Maharjan", "roll_number": "THA078BEI048", "email": "sajen.078bei048@tcioe.edu.np", "phone": "9863035364"}
    ]

class SiameseNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(10,10), padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(7,7), padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,4), padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), padding=0, stride=1),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256*6*6, out_features=4096, bias=True),
            nn.Sigmoid(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1, bias=True),
            nn.Sigmoid()
        )

    def forward_one_branch(self, x):  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)   
        x = self.fc1(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one_branch(input1)
        output2 = self.forward_one_branch(input2)
        return output1, output2

@st.cache_resource
def load_model():
    # Force CPU for compatibility 
    device = torch.device("cpu")
    model = SiameseNN()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'best_siamese_model.pth')
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)  # Ensure model is on CPU
        model.eval()
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        return None, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, device
    return model, device

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((105, 105)),
    transforms.ToTensor(),
])

def evaluate_two_images(model, img1, img2, device):
    try:
        model.eval()
        
        # Ensure both images are properly transformed and moved to device
        img1 = transform(img1).unsqueeze(0).to(device)  
        img2 = transform(img2).unsqueeze(0).to(device)
        
        # Ensure model is on the same device
        model = model.to(device)
        
        with torch.no_grad():
            output1, output2 = model(img1, img2)
            distance = nn.functional.pairwise_distance(output1, output2)
            similarity = 1 - distance.item()
            threshold = 0.5
            is_same_person = distance.item() < threshold
            
        return {
            'distance': distance.item(),
            'similarity': similarity,
            'same_person': is_same_person,
            'confidence': abs(similarity) * 100
        }
    except Exception as e:
        print(f"Error in evaluate_two_images: {e}")
        return {
            'distance': 999.0,
            'similarity': -999.0,
            'same_person': False,
            'confidence': 0.0
        }

def find_best_match(model, captured_image, device):
    best_match = None
    best_similarity = -999  # Start with very low value
    best_confidence = 0
    all_results = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    reference_dir = os.path.join(script_dir, "photos")
    
    if not os.path.exists(reference_dir):
        return None, 0, -999
    
    image_files = [f for f in os.listdir(reference_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} photos in directory")
    
    for image_file in image_files:
        try:
            ref_image_path = os.path.join(reference_dir, image_file)
            ref_image = Image.open(ref_image_path)
            
            # Compare captured image with this reference image
            result = evaluate_two_images(model, captured_image, ref_image, device)
            
            roll_number = image_file.split('.')[0]
            similarity = result['similarity']
            confidence = result['confidence']
            
            all_results.append({
                'roll_number': roll_number,
                'similarity': similarity,
                'confidence': confidence,
                'distance': result['distance']
            })
            
            print(f"Compared with {roll_number}: similarity={similarity:.4f}, confidence={confidence:.2f}%")
            
            # Keep track of the best match (highest similarity)
            if similarity > best_similarity:
                best_similarity = similarity
                best_confidence = confidence
                best_match = roll_number
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    # Sort all results by similarity (highest first)
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("\nTop 5 matches by similarity:")
    for i, result in enumerate(all_results[:5]):
        print(f"{i+1}. {result['roll_number']}: similarity={result['similarity']:.4f}, confidence={result['confidence']:.2f}%")
    
    print(f"\nBest match overall: {best_match} with similarity {best_similarity:.4f}")
    
    return best_match, best_confidence, best_similarity

def save_uploaded_image(uploaded_file, roll_number):
    if uploaded_file is not None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(script_dir, "photos")
        
        os.makedirs(images_dir, exist_ok=True)
        
        file_extension = uploaded_file.name.split('.')[-1]
        filename = f"{roll_number}.jpg"
        filepath = os.path.join(images_dir, filename)
        
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return filepath
    return None

def main():
    st.set_page_config(
        page_title="Siamese Neural Network - Face Recognition System",
        page_icon="",
        layout="wide"
    )
    
    st.title("Siamese Neural Network Face Recognition System")
    st.markdown("---")
    
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["Add Student Details", "Face Recognition", "View Students"])
    
    with tab1:
        st.header("Add New Student")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Student Information")
            name = st.text_input("Full Name", placeholder="Enter student's full name")
            roll_number = st.text_input("Roll Number", placeholder="e.g., THA078BEI049")
            email = st.text_input("Email", placeholder="student@tcioe.edu.np")
            phone = st.text_input("Phone Number", placeholder="98XXXXXXXX")
        
        with col2:
            st.subheader("Upload Photo")
            uploaded_file = st.file_uploader(
                "Choose a photo", 
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear photo of the student"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Photo", width=300)
        
        if st.button("Add Student", type="primary"):
            if all([name, roll_number, email, phone]):
                existing_student = next((s for s in st.session_state.student_database 
                                       if s['roll_number'] == roll_number), None)
                
                if existing_student:
                    st.error(f"Student with roll number {roll_number} already exists!")
                else:
                    image_path = None
                    if uploaded_file is not None:
                        image_path = save_uploaded_image(uploaded_file, roll_number)
                    
                
                    new_student = {
                        "name": name,
                        "roll_number": roll_number,
                        "email": email,
                        "phone": phone,
                        "image_path": image_path,
                        "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.student_database.append(new_student)
                    st.success(f"Student {name} added successfully!")
                    
                  
                    st.rerun()
            else:
                st.error("Please fill in all required fields!")
    
    with tab2:
        st.header("Face Recognition")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Camera Capture")
            
            camera_image = st.camera_input("Take a picture", label_visibility="hidden")
            
            if camera_image is not None:
                captured_image = Image.open(camera_image)
                
                st.image(captured_image, caption="Captured Image", width=300)
                
                if st.button("Recognize Person", type="primary"):
                    with st.spinner("Analyzing image..."):
                        best_match, confidence, similarity = find_best_match(model, captured_image, device)
                        
                        # Show debug information
                        st.write(f"**Debug Info:**")
                        st.write(f"Best match: {best_match}")
                        st.write(f"Confidence: {confidence:.2f}%")
                        st.write(f"Similarity: {similarity:.4f}")
                        
                        if best_match:  # Show result regardless of confidence
                            student_info = None
                            for student in st.session_state.student_database:
                                if best_match == student['roll_number']:
                                    student_info = student
                                    break
                            
                            if confidence > 15:  # Only show as "recognized" if confidence is reasonable
                                st.success(f"Person Recognized!")
                            else:
                                st.warning(f"Best Match Found (Low Confidence)")
                            
                            if student_info:
                                st.info(f"**Name:** {student_info['name']}")
                                st.info(f"**Roll Number:** {student_info['roll_number']}")
                                st.info(f"**Email:** {student_info['email']}")
                                st.info(f"**Phone:** {student_info['phone']}")
                            else:
                                st.info(f"**Matched Roll Number:** {best_match}")
                            
                            st.metric("Confidence Score", f"{confidence:.2f}%")
                            st.metric("Similarity Score", f"{similarity:.4f}")
                            
                        else:
                            st.error("**NO MATCH FOUND**")
                            st.warning("Unable to compare with any photos in the database.")
        
        with col2:
            st.subheader("Recognition Results")
            
            if 'recognition_history' not in st.session_state:
                st.session_state.recognition_history = []
            
            if st.session_state.recognition_history:
                st.subheader("Recent Recognitions")
                for record in st.session_state.recognition_history[-5:]:  
                    st.text(f"{record['time']}: {record['result']}")
    
    with tab3:
        st.header("Student Database")
        
        search_term = st.text_input("Search students", placeholder="Search by name, roll number, or email")
        
        filtered_students = st.session_state.student_database
        if search_term:
            filtered_students = [
                student for student in st.session_state.student_database
                if search_term.lower() in student['name'].lower() or 
                   search_term.lower() in student['roll_number'].lower() or
                   search_term.lower() in student['email'].lower()
            ]
        
        if filtered_students:
            st.subheader(f"Found {len(filtered_students)} student(s)")
            
            for i, student in enumerate(filtered_students):
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
        
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        photo_path = os.path.join(script_dir, "photos", f"{student['roll_number']}.jpg")
                        
                        if os.path.exists(photo_path):
                            try:
                                img = Image.open(photo_path)
                                st.image(img, width=120, caption=student['roll_number'])
                            except:
                                st.write("📷 No Photo")
                        elif 'image_path' in student and student['image_path'] and os.path.exists(student['image_path']):
                            try:
                                img = Image.open(student['image_path'])
                                st.image(img, width=120, caption="Uploaded")
                            except:
                                st.write("📷 No Photo")
                        else:
                            st.write("📷 No Photo")
                    
                    with col2:
                       
                        st.markdown(f"### {student['name']}")
                        
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.write(f"**Roll Number:** {student['roll_number']}")
                            st.write(f"**Email:** {student['email']}")
                        with detail_col2:
                            st.write(f"**Phone:** {student['phone']}")
                            if 'added_date' in student:
                                st.write(f"**Added:** {student['added_date']}")
                    
                    with col3:
                
                        st.write("")  
                        if os.path.exists(os.path.join(script_dir, "photos", f"{student['roll_number']}.jpg")):
                            st.success("✓ Photo Available")
                        else:
                            st.warning("⚠ No Photo")
                
                st.divider() 
        else:
            st.info("No students found matching your search criteria.")
        
        st.markdown("---")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(st.session_state.student_database))
        
        with col2:
    
            script_dir = os.path.dirname(os.path.abspath(__file__))
            photos_dir = os.path.join(script_dir, "photos")
            photo_count = 0
            if os.path.exists(photos_dir):
                for student in st.session_state.student_database:
                    photo_path = os.path.join(photos_dir, f"{student['roll_number']}.jpg")
                    if os.path.exists(photo_path):
                        photo_count += 1
            st.metric("With Photos", photo_count)
        
        with col3:
            st.metric("Displayed", len(filtered_students))
        
        with col4:
            completion_rate = (photo_count / len(st.session_state.student_database)) * 100 if st.session_state.student_database else 0
            st.metric("Photo Coverage", f"{completion_rate:.0f}%")

if __name__ == "__main__":
    main()
