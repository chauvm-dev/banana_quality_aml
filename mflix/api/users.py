from flask import Blueprint, request, jsonify
from mflix.db import create_user, find_user, get_movie, get_movies, get_movies_by_country, \
    get_movies_faceted, add_comment, update_comment, delete_comment

from flask_cors import CORS
from mflix.api.utils import expect
from datetime import datetime

users_api_v1 = Blueprint(
    'users_api_v1', 'users_api_v1', url_prefix='/api/v1/users')

CORS(users_api_v1)


@users_api_v1.route('/authentication', methods=['POST'])
def api_get_users():
    MOVIES_PER_PAGE = 20

    user_data = request.get_json()

    email = expect(user_data.get('email'), str, 'email')
    password = expect(user_data.get('password'), str, 'password')
    
    find_user(email, password)

    return jsonify(find_user(email, password))


@users_api_v1.route('register', methods=["POST"])
#@jwt_required
def api_create_user():
    """
    Posts a comment about a specific movie. Validates the user is logged in by
    ensuring a valid JWT is provided
    """
    #claims = get_jwt_claims()
    #user = User.from_claims(claims)
    user_data = request.get_json()
    try:
        email = expect(user_data.get('email'), str, 'email')
        password = expect(user_data.get('password'), str, 'password')
        full_name = expect(user_data.get('full_name'), str, 'full_name')
        
        return jsonify(create_user(email, password, full_name))
    except Exception as e:
        return jsonify({'error': str(e)}), 400