#Generate Class to read TLE data from file and save it in json format to file
from datetime import datetime, timedelta, timezone
import json
from operator import index
from TLE import TLE 
from pathlib import Path
from matplotlib import pyplot as plt


class TLEProcessor:
    def __init__(self, tle_file, json_file):
        self.tle_file = tle_file
        self.json_file = json_file
        self.tle_data = []
        self.tle_start_time: datetime = datetime.now(timezone.utc)
        self.tle_duration_hours: int = 2
        self.tle_duration_minutes: int = 0
        self.tle_end_time: datetime = self.tle_start_time + timedelta(hours=self.tle_duration_hours, minutes=self.tle_duration_minutes)

    def set_tle_start_time(self, start_time: datetime) -> None:
        self.tle_start_time = start_time
        self.tle_end_time = self.tle_start_time + timedelta(hours=self.tle_duration_hours, minutes=self.tle_duration_minutes)
        
    def set_tle_duration(self, hours: int, minutes: int=0) -> None:
        self.tle_duration_hours = hours
        self.tle_duration_minutes = minutes
        self.tle_end_time = self.tle_start_time + timedelta(hours=self.tle_duration_hours, minutes=self.tle_duration_minutes)   

    def set_tle_end_time(self, end_time: datetime) -> None:
        self.tle_end_time = end_time
        self.tle_duration_hours = int((self.tle_end_time - self.tle_start_time).total_seconds() // 3600)
        self.tle_duration_minutes = int(((self.tle_end_time - self.tle_start_time).total_seconds() % 3600) // 60)

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


    # Metod to retrieve TLE data as a list filtered by satellite name
    def get_tle_data_by_name(self, satellite_name: str=None) -> list:
        if satellite_name is None:
            return [tle for tle in self.tle_data]
        else:
            return [tle for tle in self.tle_data if tle.get_satellite_name() == satellite_name]    
        
    # Metod to retrieve TLE data as a list by index or index range
    def get_tle_data_by_index(self, start_index: int=0, end_index: int=None) -> list:
        if start_index < len(self.tle_data) and end_index is None:
            return [ self.tle_data[start_index] ]
        elif end_index is not None and start_index < end_index  and end_index <= len(self.tle_data):
            return [tle for tle in self.tle_data[start_index:end_index]]
        
        
    def plot_ground_tracks(self, satellite_names: list=None) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), gridspec_kw={'hspace': 0.3})
        for sat_name in satellite_names:
            get_tle_data = self.get_tle_data_by_name(sat_name)
            
            for tle in get_tle_data:
                # Plot ground track, latitude, longitude and altitude for each TLE object in one figure with 4 subplots
                # increase spacing between subplots
                tle.plot_ground_track(axes[0, 0])
                tle.plot_Altitude(axes[0, 1])
                tle.plot_Latitude(axes[1, 0])
                tle.plot_Longitude(axes[1, 1])

        handle, label = [], []
        for ax in axes.flatten():
            h, l = ax.get_legend_handles_labels()
            handle.extend(h)
            label.extend(l)
        
        unique_labels = dict(zip(label, handle))
        fig.legend(handles=unique_labels.values(), labels=unique_labels.keys(), loc='upper right', fontsize='small')
        plt.show()
            
    def plot_distance_between_satellites(self, satellite_name1: str, satellite_name2: str, in_view_only: bool=False) -> None:
        tle_data1 = self.get_tle_data_by_name(satellite_name1)
        tle_data2 = self.get_tle_data_by_name(satellite_name2)
        
        if not tle_data1 or not tle_data2:
            print(f"One or both satellites not found: {satellite_name1}, {satellite_name2}")
            return
        
        tle1 = tle_data1[0]
        tle2 = tle_data2[0]
        tle1.plot_distance_to_other_satellite(tle2, in_view_only=in_view_only)



if __name__ == "__main__":
    print(Path.cwd())
    tle_processor = TLEProcessor('data/starlink.txt', 'data/starlink.json')
    # Set TLE start time to current UTC time and duration to 2 hours
    tle_processor.set_tle_start_time(datetime.now(timezone.utc))
    tle_processor.set_tle_duration(hours=12, minutes=0)
    # Set global TLE start time and duration in TLE class
    TLE.TLE_START_TIME = tle_processor.tle_start_time
    TLE.TLE_END_TIME = tle_processor.tle_end_time
    tle_processor.process()
    tle_processor.plot_ground_tracks(["STARLINK-1031","STARLINK-32620","STARLINK-35057"])
    tle_processor.plot_distance_between_satellites("STARLINK-1031", "STARLINK-32620", in_view_only=False)
    