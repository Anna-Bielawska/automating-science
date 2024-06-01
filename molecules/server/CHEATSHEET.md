# Useful commands

## -1. Running server in production (for workshop participants)

First run once

```commandline
python server/start.py
```

to initialize the database, then:

```commandline
gunicorn -w 4 --worker-connections 4 --worker-class gevent 'server.start:app'
```

Best use `server/scripts/start_server.sh` helper script.

To set up ngrok from a server use

```commandline
ngrok http 8000
```

This requires setting up ngrok, for which head to the ngrok website.

## 0. Running server in development mode

```commandline
python server/start.py
```

Best use `server/tests/start_server_for_tests.sh` helper script.

## 1. Generate tokens for workshop attendees
```commandline
curl -X POST -H "Content-Type: application/json" \
     -d '{"master_key": "YourSuperSecretMasterKey"}' \
     http://127.0.0.1:5000/generate_tokens
```

## 2 Adding results

CURL:

```commandline
curl -X POST -H "Content-Type: application/json" \
-d '{
    "token": "unique_token_1",
    "metrics": {
        "metric1": 10,
        "metric2": 20
    }
}' \
http://127.0.0.1:5000/add_result

```

## 3. How to examine database

Install SQLite Browser and see `results.db` in the working directory.

## 4. How to run tests

To run tests with development server, run 

```commandline
python server/tests/test_all.py 
```

(Don't use py.test)

To run tests with production server, run 

```commandline
PORT=8000 python server/tests/test_all.py
```
## 5. How to run load tests

To run load tests, run `PYTHONPATH=`pwd` bash server/tests/load_test.sh` and then `python server/tests/load_test_summarize.py`.
