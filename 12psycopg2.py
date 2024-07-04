import psycopg2

# Define connection parameters (replace with your details)
dbname = "sankalpam_production"
user = "postgres"
password = "postgres"
host = "172.17.0.1"
port = 5432

# Connect to the database
try:
  conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
  print("Connected to database!")
except psycopg2.Error as e:
  print("Error connecting to database:", e)
  exit()
c

# Define your SQL query
sql = """
SELECT names
FROM pujas;
"""

# Create a cursor object
try:
  cur = conn.cursor()
  cur.execute(sql)
except psycopg2.Error as e:
  print("Error executing query:", e)
  conn.close()
  exit()

# Fetch data
data = cur.fetchall()

# Print the fetched data (modify this section to process data as needed)
for row in data:
  print(row)

# Close the connection
cur.close()
conn.close()

print("Successfully fetched data from database!")
