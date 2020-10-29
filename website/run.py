from config import app
from view.static_view import static_view_b

app.register_blueprint(static_view_b, url_prefix='/')

if __name__ == '__main__':
    app.run(debug=True)
