import requests


def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    userlist =  ['43086293','562121642','1871580685','241359338']
    if request.args and 'message' in request.args:
        for user in userlist:
            requests.get("https://api.telegram.org/bot5314948650:AAEWcWIqTqUKxI0YCOynQM3cJxBlAwjJ6hY/sendMessage?chat_id=" +
                         user+"&text="+str(request.args.get('message')))
        return request.args.get('message')
    elif request_json and 'message' in request_json:
        for user in userlist:
            requests.get("https://api.telegram.org/bot5314948650:AAEWcWIqTqUKxI0YCOynQM3cJxBlAwjJ6hY/sendMessage?chat_id=" +
                         user+"&text="+str(request_json['message']))
        return request_json['message']
    else:
        return f'Hello World!'
