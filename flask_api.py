from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from flask_cors import CORS
import numpy as np
import logging

# MySQL Configuration
db_config = {
    "host": "13.228.225.19",
    "user": "core3_root",
    "password": "reorio345",  # Replace with your MySQL password
    "database": "core3_bot"
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
    return None

app = Flask(__name__)

# ✅ ALLOW SPECIFIC ORIGINS
CORS(app, resources={
    r"/*": {
        "origins": ["https://core3.paradisehoteltomasmorato.com"],  # ✅ Allow Your Domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

log = logging.getLogger('werkzeug')
log.setLevel(logging.DEBUG)

# Load trained AI model
model = tf.keras.models.load_model('task_allocation_model.keras')

# Initialize tokenizer with actual vocabulary
task_vocabulary = [
    "amenities", "towels", "cleaning", "food", "room service",
    "bed sheets", "bathroom", "supplies", "breakfast", "dinner"
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(task_vocabulary)

def clean_room_number(room):
    """Extract and convert room number to integer"""
    try:
        return int(str(room).strip())
    except ValueError:
        raise ValueError(f"Invalid room number format: {room}")

# Remove the hardcoded housekeepers dictionary and add function to get employees
def get_available_employees(role):
    """Get available employees for a given role"""
    connection = get_db_connection()
    if not connection:
        return None
    
    cursor = connection.cursor(dictionary=True)
    try:
        # Debug: Check all roles in database
        cursor.execute("SELECT DISTINCT role FROM employee")
        available_roles = [r['role'] for r in cursor.fetchall()]
        log.debug(f"Available roles in database: {available_roles}")
        
        query = """
            SELECT emp_id, name, role, status
            FROM employee 
            WHERE role=%s AND status='active'
        """
        cursor.execute(query, (role,))
        employees = cursor.fetchall()
        
        # Add detailed debug logging
        log.debug(f"Query executed: {query} with role={role}")
        log.debug(f"Found employees: {employees}")
        return employees
    finally:
        cursor.close()
        connection.close()

# Function to allocate tasks
def allocate_task(task_request, task_details, room):
    connection = None
    cursor = None
    try:
        # Clean and validate room number
        room_number = clean_room_number(room)
        
        connection = get_db_connection()
        if not connection:
            log.error("Database connection failed")
            raise Exception("Database connection failed")
        
        cursor = connection.cursor()
        
        # Convert request into numerical format
        task_types = {"Request Amenities": 0, "Housekeeping": 1, "Order Food": 2}
        task_type_num = task_types.get(task_request)
        if task_type_num is None:
            raise ValueError(f"Invalid task type: {task_request}")

        # Process text input - handle details cleaning
        task_details = str(task_details).lower().strip()
        task_seq = tokenizer.texts_to_sequences([task_details])
        task_padded = pad_sequences(task_seq, maxlen=10)

        # Prepare input for model - ensure all values are numeric
        model_input = [float(task_type_num), float(room_number)]
        model_input.extend([float(x) for x in task_padded[0]])
        X_input = np.array([model_input], dtype=np.float32)

        # Debug log the input
        log.debug(f"Model input shape: {X_input.shape}, dtype: {X_input.dtype}")
        log.debug(f"Model input values: {X_input}")

        # Predict if task needs allocation
        prediction = model.predict(X_input)[0][0]

        # Update logging for prediction
        log.debug(f"Model prediction: {prediction}")

        if prediction > 0.5:
            # Update role names to match exactly what's in database
            if task_request in ["Request Amenities", "Order Food"]:
                role = "room_attendant"  # Changed from room_attendant
            else:
                role = "linen_attendant"  # Changed from linen_attendant
            
            log.debug(f"Looking for employee with role: {role}")
            available_employees = get_available_employees(role)
            
            if not available_employees:
                log.warning(f"No available {role}s found")
                return {"assigned_to": None, "status": "Pending", "message": f"No available {role}s"}
            
            # Assign to first available employee
            assigned_emp = available_employees[0]
            
            # Update employee status
            update_emp_query = """
                UPDATE employee 
                SET status='busy' 
                WHERE emp_id=%s
            """
            cursor.execute(update_emp_query, (assigned_emp['emp_id'],))
            connection.commit()

            log.debug(f"Task assigned to: {assigned_emp['name']}")
            return {
                "assigned_to": assigned_emp['name'],
                "emp_id": assigned_emp['emp_id'],
                "status": "Working"
            }
        
        log.debug("Task pending assignment")
        return {"assigned_to": None, "status": "Pending"}
    
    except ValueError as ve:
        raise ve
    except Exception as e:
        log.error(f"Task allocation failed: {str(e)}")
        raise Exception(f"Task allocation failed: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


# API Endpoint to Allocate Tasks
@app.route('/allocate-pending', methods=['GET'])
def allocate_pending_tasks():
    """Automatically allocates all pending tasks."""
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = connection.cursor(dictionary=True)

    try:
        # Fetch all pending tasks from customer_messages
        cursor.execute("""
            SELECT id, uname, request, details, room 
            FROM customer_messages 
            WHERE status='pending'
        """)
        pending_tasks = cursor.fetchall()

        if not pending_tasks:
            return jsonify({"message": "No pending tasks found"}), 200

        allocated_tasks = []
        for task in pending_tasks:
            result = allocate_task(task['request'], task['details'], task['room'])
            
            if result['assigned_to']:
                # Updated INSERT query to include task_id
                insert_query = """
                    INSERT INTO assigntasks 
                    (task_id, request, details, room, emp_name, emp_id, status) 
                    VALUES (%s, %s, %s, %s, %s, %s, 'Working')
                """
                cursor.execute(insert_query, (
                    task['id'],  # Add task_id from customer_messages
                    task['request'],
                    task['details'],
                    task['room'],
                    result['assigned_to'],
                    result['emp_id']
                ))

                # Update status in customer_messages
                update_query = """
                    UPDATE customer_messages 
                    SET status='working'
                    WHERE id=%s
                """
                cursor.execute(update_query, (task['id'],))
                connection.commit()
                
            allocated_tasks.append({
                "task_id": task["id"],
                "username": task["uname"],
                **result
            })

        return jsonify({"allocated_tasks": allocated_tasks}), 200

    except Exception as e:
        log.error(f"Error in allocation: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

# Add new endpoint to get task details
@app.route('/get-task/<task_id>', methods=['GET'])
def get_task_details(task_id):
    """Get task details including assigned employee"""
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = connection.cursor(dictionary=True)
    try:
        query = """
            SELECT task_id, emp_id, emp_name, request, details, room, status
            FROM assigntasks 
            WHERE task_id=%s AND status='Working'
        """
        cursor.execute(query, (task_id,))
        task = cursor.fetchone()

        if not task:
            return jsonify({"error": "Task not found or not in working status"}), 404

        return jsonify(task), 200

    except Exception as e:
        log.error(f"Error fetching task: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        connection.close()

@app.route('/complete-task', methods=['POST', 'OPTIONS'])
def complete_task():
    """Handle task completion and update employee status"""
    # Add OPTIONS handler for CORS preflight
    if request.method == 'OPTIONS':
        return jsonify({}), 200
        
    log.debug(f"Received complete-task request: {request.json}")
    connection = get_db_connection()
    if not connection:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = connection.cursor(dictionary=True)

    try:
        data = request.json
        if not data or 'task_id' not in data:
            return jsonify({"error": "Missing task_id"}), 400

        # First get the task details including emp_id
        query = """
            SELECT emp_id, emp_name 
            FROM assigntasks 
            WHERE task_id=%s AND status='Working'
        """
        cursor.execute(query, (data['task_id'],))
        task = cursor.fetchone()

        if not task:
            return jsonify({
                "error": "Task not found or already completed"
            }), 404

        # Update task status in both tables
        cursor.execute("""
            UPDATE assigntasks 
            SET status='Completed' 
            WHERE task_id=%s
        """, (data['task_id'],))

        cursor.execute("""
            UPDATE customer_messages 
            SET status='completed' 
            WHERE id=%s
        """, (data['task_id'],))

        # Set employee status back to active
        cursor.execute("""
            UPDATE employee 
            SET status='active' 
            WHERE emp_id=%s AND status='busy'
        """, (task['emp_id'],))

        connection.commit()

        return jsonify({
            "message": "Task completed successfully",
            "task_id": data['task_id'],
            "emp_name": task['emp_name'],
            "emp_id": task['emp_id']
        }), 200

    except Exception as e:
        log.error(f"Error completing task: {str(e)}")
        return jsonify({"error": str(e)}), 500

    finally:
        cursor.close()
        connection.close()

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "status": "success",
        "message": "API is running"
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
# End of flask_api.py