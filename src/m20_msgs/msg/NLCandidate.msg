# Version: 0.1
# One object a natural-language command could refer to (m20_nlnav).
int32 object_id                 # detector object-map id
string cls                      # detector class, e.g. "printer"
geometry_msgs/Point position    # map frame, object centre
float32 score                   # detector confidence (max over sightings)
string why                      # short reason it matched / was ranked ("nearest, 2.1 m")
