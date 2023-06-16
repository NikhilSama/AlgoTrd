#!/usr/bin/env python3

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys 
import cronitor
cronitor.api_key = '1a24dbccd8864e25862d91548ded318b'


file = open("./Data/cred.txt", "r")
keys = file.read().split()  # Get a List of keys
username = 'scaleops.ai'
password = 'EccfXCfM0EArFPft'
# Email configuration
smtp_server = 'mail.smtp2go.com'
smtp_port = 2525
sender_email = 'nikhil.sama@scaleops.ai'
receiver_email = 'nikhilsama@gmail.com'

# Create the email message
message = MIMEMultipart()
message['From'] = sender_email
message['To'] = receiver_email

def send_email(subject,body):
    message['Subject'] = subject

    body = body
    message.attach(MIMEText(body, 'plain'))

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)

def ping():
    monitor = cronitor.Monitor('AlgoTrade')

    # send a heartbeat event with a message
    monitor.ping(message="Alive!")
