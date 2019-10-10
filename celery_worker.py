from Apps import create_app, celery

app = create_app('development')
app.app_context().push()