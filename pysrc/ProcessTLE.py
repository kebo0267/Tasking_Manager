#Generate Class to read TLE data from file and save it in json format to file
from datetime import datetime, timezone
import json
from TLE import TLE 
from pathlib import Path

class TLEProcessor:
    def __init__(self, tle_file, json_file):
        self.tle_file = tle_file
        self.json_file = json_file
        self.tle_data = []

    def read_tle_data(self) -> None:
        # Implementation for reading TLE data from file
        
        with open(self.tle_file, 'r') as f:
            while True:
                # Read the three lines of TLE data
                sat_name = f.readline().strip()
                if not sat_name:  # End of file
                    break
                tle_line1 = f.readline().strip()
                tle_line2 = f.readline().strip()
                # Pass the three lines to a TLE Object for processing
                tle_object = TLE()
                tle_object.parse_tle_from_data(sat_name, tle_line1, tle_line2)
                self.tle_data.append(tle_object)
        print(f"Total TLE entries read: {len(self.tle_data)}")

    def save_to_json(self) -> None:
        # Implementation for saving data to JSON file
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump([tle.get_data_as_dict() for tle in self.tle_data], f,indent=4 )

    # Main method to process TLE data
    def process(self) -> None:
        self.read_tle_data()
        self.save_to_json()
        for tle in self.tle_data:
            print(f"Satellite: {tle.get_sat_name()}")
            start_time = datetime.now(timezone.utc)
            tle.generate_ground_track(start_time=start_time, hours=1, minutes=0)
            lat_lon_alt = tle.get_lat_lon_alt()

if __name__ == "__main__":
    print(Path.cwd())
    tle_processor = TLEProcessor('data/starlink.txt', 'data/starlink.json')
    tle_processor.process()
