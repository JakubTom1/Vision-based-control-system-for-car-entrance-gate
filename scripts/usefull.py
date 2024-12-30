import requests

def signal_of_opening(signal_state):
    """
    Sending signal for gate opening and printing ESP32 request.
    :param: Signal_stat(BOOL): State of recognition sending to ESP32.
    """
    esp_ip = "<<ESP32_IP_address>>"  # esp32 local IP address
    url = f"http://{esp_ip}/"

    # BOOL to string message
    state_value = "on" if signal_state else "off"

    try:
        response = requests.get(url, params={"state": state_value})

        # Printing response from ESP32
        if response.status_code == 200:
            print("Response ESP32:", response.text)
        elif response.status_code == 400:
            print("ESP32 Error:", response.text)  # Wrong data type
        else:
            print(f"Unexpected status code: {response.status_code}")
    except requests.RequestException as e:
        print("ESP32 connection fail:", e)

def my_license_plates():
    """
    Database of license plates of cars with entry privileges.
    :return: List of license plates in strings.
    :return: List of license plates in lists with separate characters.
    """
    myPlate1 = "<<first_license_plate_numbers>>"
    myPlate2 = "<<second_license_plate_numbers>>"
    myPlate3 = "<<third_license_plate_numbers>>"
    myPlates_list = [myPlate1, myPlate2, myPlate3]
    myPlates_lists = [list(myPlate1), list(myPlate2), list(myPlate3)]

    return myPlates_list, myPlates_lists

def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{};{};{};{}\n'.format('frame_nmr', 'license_number', 'license_number_score', 'time_of_operation'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'text' in results[frame_nmr].keys():
                    f.write('{};{};{};{}\n'.format(frame_nmr,
                                                   results[frame_nmr]['text'],
                                                   results[frame_nmr]['text_score'],
                                                   results[frame_nmr]['time'])
                            )
        f.close()