# -*- coding: utf-8 -*-
"""
Created on Sun Mar 07 2023

@author: Jet
"""

import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging
from datetime import datetime

# Change source for the adminsdk, youll find it in app/src/ within the github repo
cred = credentials.Certificate('/home/pmr/ros2_ws/src/patient_monitoring/patient_monitoring/app/pmr-test-cd8d2-firebase-adminsdk-fo4jd-05259879c0.json') 
firebase_admin.initialize_app(cred)

'''
This function sends a push notification to an android device using the PMR App.
Used by fall_detection to notify when falls occur.

Robot & device must be connected to the same network.
'''
def send_push_notification(token, message):
    timestamp = datetime.now().strftime("%Y-%m-%d at %I:%M%p")
    data = {
        'message': message,
    }

    message = messaging.Message(
        data=data,
        notification=messaging.Notification(
            title='Fall Alert',
            body=f"Patient has fallen at {timestamp}"
        ),
        token=token
    )

    try:
        response = messaging.send(message)
        print(f'Successfully sent message {message}: {response}')
    except:
        print(f"Unable to send message {message}. Check to see if your device token is valid.")
