from flask import Blueprint, request
from app.controllers.llamaindex.query_pipeline.nlp_qp import init_agent

agentController = init_agent()

api_blueprint = Blueprint("api", __name__)

@api_blueprint.route("/chat", methods=['POST'])
def llamaIndexArticleChatHandler():
  return agentController.chat(request.json)