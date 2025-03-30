from flask import Blueprint, request, jsonify, render_template, redirect, url_for
import logging
from service.chat_service import ChatService

# Blueprint oluşturulması
chat_routes = Blueprint("chat_routes", __name__)

# Loglama yapılandırması
logging.basicConfig(
    filename="logs/chat_routes.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ChatService tanımlanması
chat_service = ChatService()


@chat_routes.route("/", methods=["GET", "POST"])
def chat_page():
    """
    Kullanıcının sohbeti başlatmasını sağlayan arayüz.
    Kullanıcıdan user_id alır, bir oturum oluşturur ve chat.html'ye yönlendirir.
    """
    try:
        if request.method == "POST":
            # Kullanıcı bilgilerini al
            user_id = request.form.get("user_id", None)

            # Yeni bir oturum oluştur
            session_info = chat_service.chat_organization.start_session(user_id)
            if session_info.get("status") != "success":
                return jsonify({"status": "error", "message": "Oturum oluşturulamadı."}), 400

            session_id = session_info.get("session_id")

            # Kullanıcıyı sohbet ekranına yönlendir
            return redirect(url_for("chat_routes.chat_interface", user_id=user_id, session_id=session_id))

        # GET isteği için index ekranını döndür
        return render_template("index.html")

    except Exception as e:
        logging.error(f"Chat page yüklenirken hata oluştu: {str(e)}")
        return jsonify({"status": "error", "message": f"Sayfa yüklenemedi: {str(e)}"}), 500


@chat_routes.route("/chat", methods=["GET"])
def chat_interface():
    """
    Kullanıcının sohbet arayüzünü yükler.
    Gelen user_id ve session_id'yi alır.
    """
    try:
        user_id = request.args.get("user_id", None)
        session_id = request.args.get("session_id", None)

        # Session ID geçerliyse önceki konuşma geçmişini yükle
        validation = chat_service.chat_organization.validate_session(session_id)
        if validation.get("status") != "success":
            return jsonify({"status": "error", "message": "Geçersiz oturum. Lütfen yeniden başlayın."}), 400

        # Chat geçmişini al
        chat_history = chat_service.chat_organization.get_active_sessions().get("sessions", {}).get(session_id, [])
        return render_template(
            "chat.html",
            user_id=user_id,
            session_id=session_id,
            chat_history=chat_history
        )

    except Exception as e:
        logging.error(f"Chat interface yüklenirken hata oluştu: {str(e)}")
        return jsonify({"status": "error", "message": f"Sayfa yüklenemedi: {str(e)}"}), 500


@chat_routes.route("/api/create_session", methods=["POST"])
def create_session():
    """
    Kullanıcı için yeni bir oturum oluşturur ve session_id döner.
    """
    try:
        data = request.get_json()
        user_id = data.get("user_id", None)

        # Yeni bir oturum oluştur
        session_info = chat_service.chat_organization.start_session(user_id)
        if session_info.get("status") != "success":
            return jsonify({"status": "error", "message": "Oturum oluşturulamadı."}), 400

        session_id = session_info.get("session_id")
        return jsonify({"status": "success", "session_id": session_id}), 200

    except Exception as e:
        logging.error(f"Oturum oluşturulurken hata oluştu: {str(e)}")
        return jsonify({"status": "error", "message": f"Oturum oluşturulamadı: {str(e)}"}), 500


@chat_routes.route("/api/chat", methods=["POST"])
def chat():
    """
    Kullanıcı mesajını alır, chatbot cevabını döner ve geçmişe kaydeder.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "JSON verisi gerekli."}), 400

        session_id = data.get("session_id")
        message = data.get("message")

        if not session_id or not message:
            return jsonify({"status": "error", "message": "Session ID ve mesaj gereklidir."}), 400

        # Mesajı gönder ve model yanıtını al
        response = chat_service.send_message(session_id, message)
        if response.get("status") == "error":
            return jsonify(response), 400

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "response": response.get("response")
        }), 200

    except Exception as e:
        logging.error(f"Chat API çağrısı başarısız oldu: {str(e)}")
        return jsonify({"status": "error", "message": f"Hata oluştu: {str(e)}"}), 500


@chat_routes.route("/api/get_chat_history", methods=["GET"])
def get_chat_history():
    """
    Belirli bir oturuma ait sohbet geçmişini döner.
    """
    try:
        session_id = request.args.get("session_id")
        if not session_id:
            return jsonify({"status": "error", "message": "Session ID gereklidir."}), 400

        # Sohbet geçmişini al
        chat_history = chat_service.chat_organization.get_active_sessions().get("sessions", {}).get(session_id, [])
        return jsonify({"status": "success", "chat_history": chat_history}), 200

    except Exception as e:
        logging.error(f"Sohbet geçmişi alınamadı: {str(e)}")
        return jsonify({"status": "error", "message": f"Sohbet geçmişi alınamadı: {str(e)}"}), 500
