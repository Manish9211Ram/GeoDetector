web: sh -c 'python manage.py collectstatic --noinput && python manage.py migrate && gunicorn GeoDetector.wsgi:application --bind 0.0.0.0:$PORT'
