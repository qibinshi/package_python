import os
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

if not os.path.exists("complete_mag_catalog"):
    os.makedirs("complete_mag_catalog")

client = Client("ISC")
starttime = UTCDateTime("2000-01-01")
endtime = UTCDateTime("2021-12-31")
cat = client.get_events(starttime=starttime,
                        endtime=endtime,
                        minmagnitude=5.5,
                        includeallmagnitudes=True,
                        mindepth=100.0,
                        catalog="ISC")
print(len(cat), "are found")
for ev in cat:
    filename = "complete_mag_catalog/" + str(ev.resource_id)[13:] + ".xml"
    ev.write(filename, format="QUAKEML")