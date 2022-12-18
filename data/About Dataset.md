# Datasets Information

## Dataset of the experts

### Features

| name           | note                                                          |
|----------------|---------------------------------------------------------------|
| file_name      | name of the image file                                        |
| road           | calculated FoV of the road on the image                       |
| sidewalk       | calculated FoV of the sidewalk on the image                   |
| building       | calculated FoV of the building on the image                   |
| wall           | calculated FoV of the wall on the image                       |
| fence          | calculated FoV of the fence on the image                      |
| pole           | calculated FoV of the pole on the image                       |
| traffic_light  | calculated FoV of the traffic light on the image              |
| traffic_sign   | calculated FoV of the traffic sign on the image               |
| vegetation     | calculated FoV of the vegetation on the image                 |
| terrain        | calculated FoV of the terrain on the image                    |
| sky            | calculated FoV of the sky on the image                        |
| person         | calculated FoV of the person on the image                     |
| rider          | calculated FoV of the bicycle rider on the image              |
| car            | calculated FoV of the car on the image                        |
| truck          | calculated FoV of the truck on the image                      |
| bus            | calculated FoV of the bus on the image                        |
| train          | calculated FoV of the train on the image                      |
| motorcycle     | calculated FoV of the motorcycle on the image                 |
| bicycle        | calculated FoV of the bicycle on the image                    |
| visual_entropy | the visual entropy of the image (based on the calculated FoV) |
| greenery       | calculated FoV of the greenary on the image                   |
| car_num        | the number of cars on the image                               |
| people_num     | the number of people on the image                             |
| pole_num       | the number of poles on the image                              |
| isline         | is any wire on the image or not (1: yes 0: no)                |
| streetlamp_num | the number of street lamps on the image                       |
| is_safe        | safety annotation of the image (1: safe 0: unsafe)            |
| complexity     | Normalization of feature-weighted sums                        |



## Dataset of the crowdsourcing annotations

### Features

| name           | note                                                         |
|----------------| ------------------------------------------------------------ |
| file_name      | name of the image file                                       |
| road           | calculated FoV of the road on the image                      |
| sidewalk       | calculated FoV of the sidewalk on the image                  |
| building       | calculated FoV of the building on the image                  |
| wall           | calculated FoV of the wall on the image                      |
| fence          | calculated FoV of the fence on the image                     |
| pole           | calculated FoV of the pole on the image                      |
| traffic_light  | calculated FoV of the traffic light on the image             |
| traffic_sign   | calculated FoV of the traffic sign on the image              |
| vegetation     | calculated FoV of the vegetation on the image                |
| terrain        | calculated FoV of the terrain on the image                   |
| sky            | calculated FoV of the sky on the image                       |
| person         | calculated FoV of the person on the image                    |
| rider          | calculated FoV of the bicycle rider on the image             |
| car            | calculated FoV of the car on the image                       |
| truck          | calculated FoV of the truck on the image                     |
| bus            | calculated FoV of the bus on the image                       |
| train          | calculated FoV of the train on the image                     |
| motorcycle     | calculated FoV of the motorcycle on the image                |
| bicycle        | calculated FoV of the bicycle on the image                   |
| visual_entropy | the visual entropy of the image (based on the calculated FoV) |
| greenery       | calculated FoV of the greenary on the image                  |
| score          | score computed using TrueSkill                               |
| all_pred       | score computed using LambdaRank and TrueSkill                |
| ranked_level   | rank of images based on the all_pred                         |
| complexity     | Normalization of feature-weighted sums                        |
