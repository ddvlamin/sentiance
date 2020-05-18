from unittest.mock import patch
from unittest import TestCase, main

import pandas as pd

from sentiance.location import *

class TestUserLocations(TestCase):

  def setUp(self):
    self.user_locations_df = pd.DataFrame([
      { "latitude": 51.17165565490723, "longitude": 4.346981048583984},
      { "latitude": 51.1720, "longitude": 4.3476 },
      { "latitude": 51.1716, "longitude": 4.3459 }
    ])

  @patch("sentiance.location.pd.read_csv")
  def test__contains__(self, mock_read_csv):
    mock_read_csv.return_value = self.user_locations_df
    locations = UserLocations("testfile", 36)
    self.assertEqual(len(locations), 2)

if __name__ == "__main__":
  main()
