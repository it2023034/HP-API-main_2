import csv
import re

def convert_to_csv(input_filename: str, output_filename: str):
    message_pattern = re.compile(r'(?:\\s*)?\[(\d{2}/\d{2}/\d{4},\s*\d{2}:\d{2})\]\s*(.*?):\s*(.*)')
    continuation_pattern = re.compile(r'^\\s*(.*)')
 
    parsed_data = []
    participants = set()
    current_message = None

    with open(input_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('Participants:') or line.startswith('Date range:') or line.startswith('---') or 'Whatsapp chat history' in line:
                continue

            match = message_pattern.match(line)
            if match:
                if current_message:
                    parsed_data.append(current_message)
                
                time_str, sender, text = match.groups()
                participants.add(sender)
                
                current_message = {
                    'Time': time_str,
                    'Sender': sender,
                    'Message': text
                }
            else:
                if current_message:
                    cont_match = continuation_pattern.match(line)
                    added_text = cont_match.group(1) if cont_match else line
                    
                    if added_text:
                        current_message['Message'] += ' ' + added_text

    if current_message:
        parsed_data.append(current_message)

    participants_list = list(participants)

    with open(output_filename, 'w', encoding='utf-8', newline='') as csvfile:
        # 1. Δημιουργούμε τον writer για τα headers (QUOTE_MINIMAL)
        header_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        header_writer.writerow(['Time', 'Sender', 'Receiver', 'Message'])
        
        # 2. Δημιουργούμε έναν ΝΕΟ writer για τα δεδομένα (QUOTE_ALL) στον ίδιο φάκελο/αρχείο
        data_writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        
        for row in parsed_data:
            sender = row['Sender']
            receiver = participants_list[1] if sender == participants_list[0] else participants_list[0]
            
            # Γράφουμε τα δεδομένα χρησιμοποιώντας τον data_writer
            data_writer.writerow([row['Time'], sender, receiver, row['Message']])
            
    return True