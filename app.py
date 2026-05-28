from flask import Flask, request, jsonify
import os
import io
import base64
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import shutil
from mvector.predict import MVectorPredictor
from loguru import logger
import traceback

app = Flask(__name__)

# 配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'wma'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局预测器实例
predictor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_predictor():
    """初始化预测器"""
    global predictor
    try:
        predictor = MVectorPredictor(
            configs='configs/cam++.yml',
            threshold=0.6,
            audio_db_path='audio_db',
            model_path='models/CAMPPlus_Fbank/best_model/',
            use_gpu=True,
            log_level="info"
        )
        logger.info("预测器初始化成功")
    except Exception as e:
        logger.error(f"预测器初始化失败: {str(e)}")
        predictor = None

def get_audio_from_request():
    """从请求中获取音频数据"""
    # 方式1: 上传文件
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return filepath, 'file'
    
    # 方式2: Base64编码的音频数据
    if 'audio_data' in request.form:
        try:
            audio_data = request.form['audio_data']
            audio_bytes = base64.b64decode(audio_data)
            return audio_bytes, 'bytes'
        except Exception as e:
            logger.error(f"解析base64音频数据失败: {str(e)}")
    
    # 方式3: JSON中的音频数据
    if request.is_json:
        data = request.get_json()
        if 'audio_data' in data:
            try:
                audio_data = data['audio_data']
                audio_bytes = base64.b64decode(audio_data)
                return audio_bytes, 'bytes'
            except Exception as e:
                logger.error(f"解析JSON音频数据失败: {str(e)}")
    
    return None, None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'predictor_ready': predictor is not None,
        'message': '声纹识别API运行正常'
    })

@app.route('/init', methods=['POST'])
def initialize():
    """初始化预测器"""
    try:
        data = request.get_json() or {}
        
        # 可选参数
        configs = data.get('configs', 'configs/cam++.yml')
        threshold = data.get('threshold', 0.8)
        audio_db_path = data.get('audio_db_path', 'audio_db')
        model_path = data.get('model_path', 'models/CAMPPlus_Fbank/best_model/')
        use_gpu = data.get('use_gpu', True)
        log_level = data.get('log_level', 'info')
        
        global predictor
        predictor = MVectorPredictor(
            configs=configs,
            threshold=threshold,
            audio_db_path=audio_db_path,
            model_path=model_path,
            use_gpu=use_gpu,
            log_level=log_level
        )
        
        return jsonify({
            'status': 'success',
            'message': '预测器初始化成功',
            'config': {
                'configs': configs,
                'threshold': threshold,
                'audio_db_path': audio_db_path,
                'model_path': model_path,
                'use_gpu': use_gpu,
                'log_level': log_level
            }
        })
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'初始化失败: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """预测音频特征"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': '预测器未初始化'}), 400
    
    try:
        audio_data, data_type = get_audio_from_request()
        if audio_data is None:
            return jsonify({'status': 'error', 'message': '未找到音频数据'}), 400
        
        # 获取可选参数
        sample_rate = int(request.form.get('sample_rate', 16000))
        
        # 预测
        feature = predictor.predict(audio_data, sample_rate=sample_rate)
        
        # 清理临时文件
        if data_type == 'file' and os.path.exists(audio_data):
            os.remove(audio_data)
        
        return jsonify({
            'status': 'success',
            'feature': feature.tolist(),
            'feature_dim': len(feature)
        })
    
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'status': 'error', 'message': f'预测失败: {str(e)}'}), 500

@app.route('/contrast', methods=['POST'])
def contrast():
    """声纹对比"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': '预测器未初始化'}), 400
    
    try:
        # 获取两个音频文件
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'status': 'error', 'message': '需要上传两个音频文件'}), 400
        
        file1 = request.files['file1']
        file2 = request.files['file2']
        
        if not (file1 and file1.filename and allowed_file(file1.filename) and
                file2 and file2.filename and allowed_file(file2.filename)):
            return jsonify({'status': 'error', 'message': '音频文件格式不支持'}), 400
        
        # 保存临时文件
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], f"temp1_{filename1}")
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], f"temp2_{filename2}")
        
        file1.save(filepath1)
        file2.save(filepath2)
        
        # 声纹对比
        similarity = predictor.contrast(filepath1, filepath2)
        similarity = float(similarity) #显式转换float类型

        is_same_person = bool(similarity >= predictor.threshold)
        
        # 清理临时文件
        os.remove(filepath1)
        os.remove(filepath2)
        
        return jsonify({
            'status': 'success',
            'similarity': float(similarity),
            'is_same_person': is_same_person
        })
    
    except Exception as e:
        logger.error(f"声纹对比失败: {str(e)}")
        return jsonify({'status': 'error', 'message': f'声纹对比失败: {str(e)}'}), 500

@app.route('/register', methods=['POST'])
def register():
    """声纹注册"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': '预测器未初始化'}), 400
    
    try:
        audio_data, data_type = get_audio_from_request()
        if audio_data is None:
            return jsonify({'status': 'error', 'message': '未找到音频数据'}), 400
        
        # 获取用户名
        user_name = request.form.get('user_name') or (request.get_json() or {}).get('user_name')
        if not user_name:
            return jsonify({'status': 'error', 'message': '用户名不能为空'}), 400
        
        # 获取可选参数
        sample_rate = int(request.form.get('sample_rate', 16000))
        
        # 注册
        success, message = predictor.register(audio_data, user_name, sample_rate=sample_rate)
        
        # 清理临时文件
        if data_type == 'file' and os.path.exists(audio_data):
            os.remove(audio_data)
        
        return jsonify({
            'status': 'success' if success else 'error',
            'message': message,
            'user_name': user_name
        })
    
    except Exception as e:
        logger.error(f"声纹注册失败: {str(e)}")
        return jsonify({'status': 'error', 'message': f'声纹注册失败: {str(e)}'}), 500

@app.route('/recognition', methods=['POST'])
def recognition():
    """声纹识别"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': '预测器未初始化'}), 400
    
    try:
        audio_data, data_type = get_audio_from_request()
        if audio_data is None:
            return jsonify({'status': 'error', 'message': '未找到音频数据'}), 400
        
        # 获取可选参数
        threshold = request.form.get('threshold')
        if threshold:
            threshold = float(threshold)
        sample_rate = int(request.form.get('sample_rate', 16000))
        
        # 识别
        result = predictor.recognition(audio_data, threshold=threshold, sample_rate=sample_rate)
        
        # 清理临时文件
        if data_type == 'file' and os.path.exists(audio_data):
            os.remove(audio_data)
        
        return jsonify({
            # 'status': 'success',
            'user_name': result[0],
            'similarity': result[1],
            'recognized': result[0] is not None
        })
    
    except Exception as e:
        logger.error(f"声纹识别失败: {str(e)}")
        return jsonify({'status': 'error', 'message': f'声纹识别失败: {str(e)}'}), 500

@app.route('/users', methods=['GET'])
def get_users():
    """获取所有用户"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': '预测器未初始化'}), 400
    
    try:
        users = predictor.get_users()
        unique_users = list(set(users))
        
        return jsonify({
            'status': 'success',
            'users_name': unique_users,
            'total_count': len(unique_users),
            'total_samples': len(users)
        })
    
    except Exception as e:
        logger.error(f"获取用户列表失败: {str(e)}")
        return jsonify({'status': 'error', 'message': f'获取用户列表失败: {str(e)}'}), 500

@app.route('/users/delete_user', methods=['DELETE'])
def remove_user():
    """删除用户"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': '预测器未初始化'}), 400
    
    try:
        # 获取请求体中的 JSON 数据
        data = request.get_json()
        if not data or 'user_name' not in data:
            return jsonify({'status': 'error', 'message': '请求体中缺少 user_name 字段'}), 400
        
        # 从请求体中提取用户名
        user_name = data['user_name']
        success = predictor.remove_user(user_name)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'用户 {user_name} 删除成功'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'用户 {user_name} 不存在'
            }), 404
    
    except Exception as e:
        logger.error(f"删除用户失败: {str(e)}")
        return jsonify({'status': 'error', 'message': f'删除用户失败: {str(e)}'}), 500

@app.route('/speaker_diarization', methods=['POST'])
def speaker_diarization():
    """说话人日志识别"""
    if predictor is None:
        return jsonify({'status': 'error', 'message': '预测器未初始化'}), 400
    
    try:
        audio_data, data_type = get_audio_from_request()
        if audio_data is None:
            return jsonify({'status': 'error', 'message': '未找到音频数据'}), 400
        
        # 获取可选参数
        sample_rate = int(request.form.get('sample_rate', 16000))
        speaker_num = request.form.get('speaker_num')
        if speaker_num:
            speaker_num = int(speaker_num)
        search_audio_db = request.form.get('search_audio_db', 'false').lower() == 'true'
        
        # 说话人日志识别
        result = predictor.speaker_diarization(
            audio_data, 
            sample_rate=sample_rate, 
            speaker_num=speaker_num,
            search_audio_db=search_audio_db
        )
        
        # 清理临时文件
        if data_type == 'file' and os.path.exists(audio_data):
            os.remove(audio_data)
        
        return jsonify({
            'status': 'success',
            'segments': result,
            'total_segments': len(result)
        })
    
    except Exception as e:
        logger.error(f"说话人日志识别失败: {str(e)}")
        return jsonify({'status': 'error', 'message': f'说话人日志识别失败: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'status': 'error', 'message': '文件太大，最大支持16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': 'API接口不存在'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'status': 'error', 'message': '服务器内部错误'}), 500

if __name__ == '__main__':
    # 启动时初始化预测器
    init_predictor()
    
    # 启动Flask应用
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
        threaded=True
    )
