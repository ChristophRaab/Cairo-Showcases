

import requests
url ="https://api.spotify.com/v1/search?q=Guillaume%20Dufay%09Missa%20L'homme%20arme%3A%20Kyrie%09Dufay%3A%20Missa%20L'%20Homme%20Arme&type=track"
type = "type=track"

data = '''{
 "Accept": "application/json",
 "Authorization": "BearerQB2XtO3511y-gGdk506AhtjVnjvjfjPicaCQiCcqMTpkWX-qIaK8aw4meqbdGQR2kHEW-ufb7MZOMFG4-7HkG0gheDOHK4Na7AH6ZcSBnol3q6zTMP9Y2qQaHiFnZqlwxMyb9QhHRixuAY"
}'''
response = requests.post(url, data=data)
print(response)