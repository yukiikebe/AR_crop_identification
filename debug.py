# import streamlit as st
# from datetime import datetime, timedelta

# # Title for your app
# st.title('Date Range Selection in Streamlit')

# # Create a start date input widget
# start_date = st.date_input("Select a start date:", datetime.today())

# # Create an end date input widget
# # By default, set it to one week from the start date
# end_date = st.date_input("Select an end date:", start_date + timedelta(days=7))

# # Check if the start date is after the end date and display a warning
# if start_date > end_date:
#     st.error('Error: End date must be after start date.')

# # Display the selected date range
# st.write('You selected:', start_date, 'to', end_date)

import streamlit as st

# Dummy function to represent the download process
def download_data():
    # Your download logic here
    st.write("Data is being downloaded...")

# Dummy function to inform the user
def inform_user_process_completed():
    st.write("Data has already been downloaded.")

# Check if the download has been done
if 'is_downloaded' not in st.session_state:
    st.session_state['is_downloaded'] = False

if not st.session_state['is_downloaded']:
    # Perform the download
    download_data()
    # Update the session state
    st.session_state['is_downloaded'] = True
else:
    # Inform the user
    inform_user_process_completed()

print("Done")