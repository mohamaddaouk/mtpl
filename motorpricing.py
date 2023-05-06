# Author    : Mohamad Kheir EL Daouk
# Created on: Sat May 06 18:28:20 2023

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import json

from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu

from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)
import cufflinks as cf
cf.go_offline()

st.set_page_config(layout = 'wide')


#-----------------------------------------------------------------------------#
# Main Title
st.markdown(f"""
        <h1>
            <h1 style="vertical-align:center;font-size:55px;padding-left:50px;color:#00769A;padding-top:0px;margin-left:0em";>
            Introducing Machine Learning for Motor Third Party Liability Pricing in the Lebanese Insurance Industry
        </h1>""",unsafe_allow_html = True)
#-----------------------------------------------------------------------------#
# Menu bar
selected = option_menu(
    menu_title = None,
    options = ["  Home  ", "  Pricing Engine  "],    
    menu_icon = "cast",
    icons = ['house', 'calculator'], 
    default_index = 0,
    orientation = "horizontal",
    styles = {"nav-link-selected":{"background-color":"#00769A"}}
)
#-----------------------------------------------------------------------------#
# Function to load animations
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
#-----------------------------------------------------------------------------#
# Import Models
@st.cache(allow_output_mutation=True) 
def load_models():
    private_freq = joblib.load("Best_model_Optimized_RandomForestRegressor_Private_Frequency_model_cap3k.pkl")
    private_sev = joblib.load("Best_model_Optimized_XGBRegressor_Private_Severity_model_cap3k.pkl")  
    commercial_freq = joblib.load("Best_model_Optimized_XGboost_Commercial_Frequency_model_cap3k.pkl") 
    commercial_sev = joblib.load("Best_model_Optimized_XGBRegressor_Commercial_Severity_model_cap3k.pkl")
    
    return private_freq, private_sev, commercial_freq, commercial_sev

# Load the Models
private_freq, private_sev, commercial_freq, commercial_sev = load_models() 


def calculate_age(date_of_birth):
    today = datetime.date.today()
    policy_holder_age = today.year - date_of_birth.year
    return policy_holder_age


#-----------------------------------------------------------------------------#
# Component 1: Home
if selected == "  Home  ":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
                <h2>
                    <h4 style="vertical-align:center;font-size:28px;color:#00769A;padding-left:0px;padding-top:5px;margin-left:0em";>
                    Drive with peace of mind, we've got you covered
                </h2>""",unsafe_allow_html = True)


    with col2:
        coding_lottie = load_lottiefile("home.json")
        st_lottie(coding_lottie,
        speed = 1,
        reverse = False,
        loop = True,
        quality = 'high',
        height = 300,
        width = 700,
        key = None
        )

 #-----------------------------------------------------------------------------#
# Component 2: Pricing Engine
if selected == "  Pricing Engine  ":
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Personal Information")
        # collecting personal info
        first_name = st.text_input('First Name')
        last_name = st.text_input('Last Name')
        
        #Initalize
        date_of_birth = datetime.date(2005, 4, 4)
        gender = ' '
        primary_residence = ' ' 
        
        if len(first_name) > 1 and len(last_name) > 1:
            st.header(first_name.title() + ', tell us more about yourself')
            date_of_birth = st.date_input("Select Your Date of Birth", datetime.date(2005, 4, 4), key='date_of_birth_input')
            if calculate_age(date_of_birth) <18:
                st.warning("Age should be at leat 18 years!")
                # Set focus on the date_of_birth field
                script = """
                <script>
                    document.getElementById('date_of_birth_input').focus();
                </script>
                """
                st.markdown(script, unsafe_allow_html=True)
                           
            gender = st.selectbox('Gender', (' ', 'Male', 'Female', 'Other'))

            primary_residence = st.selectbox('Primary Residence Area', (' ', 'Aakkar', 'Baalbek-Hermel', 
            'Beqaa', 'Beirut', 'Mount Lebanon', 'Nabatiyeh', 'North Lebanon', 'South Lebanon'), key='primary_residence_input')

    with col2:
        # collecting vehicle info
        st.header('Tell us about your vehicle')
        vehicle_year_build = st.number_input('Year', min_value = 1990 ,max_value = 2023, value=2019)
        
        vehicle_make = st.selectbox('Make', (
            ' ', 
            'AC', 'Acura', 'Alfa Romeo', 'Alpina', 'Ariel', 'Ascari', 'Aston Martin', 'Audi', 'Beijing', 'Bentley', 'Bizzarrini', 
            'BMW', 'Bristol', 'Bugatti', 'Buick', 'Cadillac', 'Caterham', 'Chevrolet', 'Chrysler', 'Citroen', 'Daewoo',
            'Daihatsu', 'De Tomaso', 'Dodge', 'Donkervoort', 'Eagle', 'Ferrari', 'Fiat', 'Ford', 'GAZ', 'Ginetta',
            'GMC', 'Holden', 'Honda', 'Hummer', 'Hyundai', 'Infiniti', 'Isuzu', 'Jaguar', 'Jeep', 'Jensen', 'Kia',
            'Lada', 'Lamborghini', 'Lancia', 'Land Rover', 'Lexus', 'Lincoln', 'Lotec', 'Lotus', 'Mahindra', 'Marcos',
            'Matra-Simca', 'Mazda', 'MCC', 'Mercedes-Benz', 'Mercury', 'Mini', 'Mitsubishi', 'Morgan', 'Nissan', 'Noble',
            'Oldsmobile', 'Opel', 'Pagani', 'Panoz', 'Peugeot', 'Pininfarina', 'Plymouth', 'Pontiac', 'Porsche',
            'Proton', 'Renault' ,'Riley', 'Rolls-Royce', 'Rover', 'Saab', 'Saleen', 'Samsung', 'Saturn', 'Seat', 
            'Skoda', 'Smart', 'SsangYong', 'Subaru', 'Suzuki', 'Tata', 'Toyota', 'TVR', 'Vauxhall', 'Vector', 'Venturi',
            'Volkswagen', 'Volvo', 'Westfield', 'ZAZ', '- Other -'))
            
        vehicle_model = st.text_input('Model')
        
        body_type = st.selectbox('Body Type', (' ', 'Sedan', 'Coupe', 'Sports Car', 'Station Wagon', 'Hatchback', 'Convertible',
        'Sport Utility Vehicle (SUV)', 'Minivan', 'Pickup Truck'))

        horsepower = st.number_input('Horse Power', min_value = 4 ,max_value = 500, value=17)
        
        seatcapacity = st.selectbox('Seat Capacity', (' ', '2', '4', '5', '6', '7', '8+'))
        
        primary_use = st.selectbox('Primary Use', ('', 'Private', 'Commercial'))



    with col3:
        st.write(' ')
        if st.button('Get a Quotation'):
            try:
                policy_holder_age = calculate_age(date_of_birth)
            except:
                pass

            vehicle_age = datetime.date.today().year - vehicle_year_build
            
            PopulationDensity = {'Aakkar':537.558375634518,
                     'Beirut':21881.2626262626,
                     'Beqaa':372.883461270063,
                     'Mount Lebanon':1227.79967689822,
                     'Nabatiyeh':362.796786389414,
                     'North Lebanon':652.655239327296,
                     'South Lebanon':634.765490533563,
                     'Baalbek-Hermel':138.393818544367}


            try:
                PopDensityError = False
                PopulationDensity = PopulationDensity[primary_residence]
            except:
                PopDensityError = True
            
            
            
            IsError = True
            if (len(first_name) <= 1) | (len(last_name) <= 1):
                error_message = "Please enter your first and last name."
            elif date_of_birth is None:
                error_message = "Please enter your name and then enter your date of birth"
            elif calculate_age(date_of_birth) <18:
                error_message = "Age should be at leat 18 years!"
            elif calculate_age(date_of_birth) >=100:
                error_message = "Please refer to the company for a quotation" #"Age is over 100 years!"
            elif gender == ' ':
                 error_message = "Please select gender."
            elif (primary_residence == " ") | PopDensityError:
                error_message = "Please enter primary residence."
            elif primary_use == "":
                error_message = "Please enter primary usage of Vehicle."
            else:
                IsError = False
            
            
            if IsError:
                st.write(error_message)
            else:
                my_X = pd.DataFrame({'InsuredAge':[policy_holder_age], 'VehicleAge': [vehicle_age], 'HorsePower':[horsepower], 'PopulationDensity':[PopulationDensity]})
                try:
                    if primary_use == "Private":
                        freq_predictions = private_freq.predict(my_X)
                        sev_predictions = private_sev.predict(my_X)
                    
                    elif primary_use == "Commercial":
                        freq_predictions = commercial_freq.predict(my_X)
                        sev_predictions = commercial_sev.predict(my_X)
                    else:
                        freq_predictions = 0; sev_predictions = 0
                    if (freq_predictions == 0) & (sev_predictions == 0):
                        st.write("Please refer to the company for a quotation.")
                    else:
                        quotation = round(((np.maximum(5, freq_predictions[0]*sev_predictions[0])*1.55+12)*1.11+1.33) * 1.023**5 *.67) # as at 2023 and adjusted for inflation
                        st.header('Your policy premium is')
                        st.title('$' + str(quotation))
                except Exception as e:
                  st.write("Please refer to the company for a quotation")  


            coding_lottie = load_lottiefile("car.json")
            st_lottie(coding_lottie,
            speed = 1,
            reverse = False,
            loop = True,
            quality = 'high',
            height = 300,
            width = 400,
            key = None
            )

        