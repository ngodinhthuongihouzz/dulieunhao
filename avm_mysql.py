import mysql.connector

import logging
import os
import cloudstorage as gcs
import webapp2

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password=""
)

print(mydb)

