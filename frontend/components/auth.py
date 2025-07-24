import streamlit as st
import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import secrets
import re

def check_authentication() -> bool:
    """Main authentication check function"""
    
    # Initialize authentication state
    initialize_auth_state()
    
    # Check if authentication is disabled (for development)
    if st.session_state.get('auth_disabled', False):
        return True
    
    # Check if user is already authenticated
    if is_user_authenticated():
        return True
    
    # Show login interface
    return render_login_interface()

def initialize_auth_state():
    """Initialize authentication-related session state"""
    
    if 'auth_initialized' not in st.session_state:
        st.session_state.auth_initialized = True
        st.session_state.user_authenticated = False
        st.session_state.current_user = None
        st.session_state.login_attempts = 0
        st.session_state.last_login_attempt = None
        st.session_state.session_token = None
        st.session_state.auth_disabled = False  # Set to True for development
        
        # Initialize user database (in production, use proper database)
        if 'user_database' not in st.session_state:
            st.session_state.user_database = get_default_users()

def get_default_users() -> Dict[str, Dict[str, Any]]:
    """Get default user accounts (for demo purposes)"""
    
    return {
        "admin": {
            "password_hash": hash_password("admin123"),
            "role": "admin",
            "email": "admin@aiethics.dev",
            "full_name": "System Administrator",
            "created_at": datetime(2024, 1, 1),
            "last_login": None,
            "permissions": ["read", "write", "delete", "admin"],
            "active": True
        },
        "analyst": {
            "password_hash": hash_password("analyst123"),
            "role": "analyst",
            "email": "analyst@aiethics.dev", 
            "full_name": "Ethics Analyst",
            "created_at": datetime(2024, 1, 1),
            "last_login": None,
            "permissions": ["read", "write"],
            "active": True
        },
        "viewer": {
            "password_hash": hash_password("viewer123"),
            "role": "viewer",
            "email": "viewer@aiethics.dev",
            "full_name": "Report Viewer",
            "created_at": datetime(2024, 1, 1),
            "last_login": None,
            "permissions": ["read"],
            "active": True
        }
    }

def render_login_interface() -> bool:
    """Render the login interface"""
    
    # Clear the main content area for login
    st.markdown("""
    <style>
    .main .block-container {
        max-width: 400px;
        margin-top: 5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Login form container
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🛡️</div>
            <h1 style="color: #2E5AAC; margin-bottom: 0.5rem;">AI Ethics Toolkit</h1>
            <p style="color: #64748B;">Please sign in to continue</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check for rate limiting
        if is_rate_limited():
            st.error("🚫 Too many login attempts. Please wait before trying again.")
            return False
        
        # Login tabs
        tab1, tab2 = st.tabs(["🔑 Sign In", "📝 Demo Accounts"])
        
        with tab1:
            return render_login_form()
        
        with tab2:
            render_demo_accounts_info()
            return False

def render_login_form() -> bool:
    """Render the login form"""
    
    with st.form("login_form", clear_on_submit=False):
        st.markdown("#### Sign In")
        
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            help="Use 'admin', 'analyst', or 'viewer' for demo"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            help="Demo passwords: admin123, analyst123, viewer123"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            remember_me = st.checkbox("Remember me")
        
        with col2:
            forgot_password = st.button("Forgot Password?", type="secondary")
        
        login_clicked = st.form_submit_button("🔑 **Sign In**", type="primary", use_container_width=True)
        
        if login_clicked:
            return handle_login_attempt(username, password, remember_me)
        
        if forgot_password:
            handle_forgot_password()
    
    return False

def render_demo_accounts_info():
    """Render demo accounts information"""
    
    st.markdown("#### 🧪 Demo Accounts")
    st.info("Use these accounts to explore different permission levels:")
    
    demo_accounts = [
        {
            "username": "admin",
            "password": "admin123",
            "role": "Administrator",
            "permissions": "Full access to all features",
            "icon": "👑"
        },
        {
            "username": "analyst", 
            "password": "analyst123",
            "role": "Ethics Analyst",
            "permissions": "Can run audits and generate reports",
            "icon": "📊"
        },
        {
            "username": "viewer",
            "password": "viewer123", 
            "role": "Report Viewer",
            "permissions": "Read-only access to reports",
            "icon": "👁️"
        }
    ]
    
    for account in demo_accounts:
        st.markdown(f"""
        <div style="
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            background: #F8FAFC;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 1.5rem; margin-right: 8px;">{account['icon']}</span>
                <strong>{account['role']}</strong>
            </div>
            <div style="font-family: monospace; font-size: 0.9rem; margin-bottom: 4px;">
                <strong>Username:</strong> {account['username']}<br>
                <strong>Password:</strong> {account['password']}
            </div>
            <div style="font-size: 0.85rem; color: #64748B;">
                {account['permissions']}
            </div>
        </div>
        """, unsafe_allow_html=True)

def handle_login_attempt(username: str, password: str, remember_me: bool) -> bool:
    """Handle login attempt"""
    
    # Validate input
    if not username or not password:
        st.error("❌ Please enter both username and password")
        return False
    
    # Update login attempt tracking
    st.session_state.login_attempts += 1
    st.session_state.last_login_attempt = datetime.now()
    
    # Authenticate user
    user_data = authenticate_user(username, password)
    
    if user_data:
        # Successful login
        st.session_state.user_authenticated = True
        st.session_state.current_user = {
            'username': username,
            'role': user_data['role'],
            'email': user_data['email'],
            'full_name': user_data['full_name'],
            'permissions': user_data['permissions'],
            'login_time': datetime.now()
        }
        
        # Generate session token
        st.session_state.session_token = generate_session_token(username)
        
        # Update last login
        st.session_state.user_database[username]['last_login'] = datetime.now()
        
        # Reset login attempts
        st.session_state.login_attempts = 0
        
        # Show success message and rerun
        st.success(f"✅ Welcome back, {user_data['full_name']}!")
        st.rerun()
        
    else:
        # Failed login
        st.error("❌ Invalid username or password")
        
        # Lock account after too many attempts
        if st.session_state.login_attempts >= 5:
            st.error("🚫 Account temporarily locked due to multiple failed attempts")
        
        return False

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials"""
    
    user_db = st.session_state.get('user_database', {})
    
    if username not in user_db:
        return None
    
    user_data = user_db[username]
    
    # Check if account is active
    if not user_data.get('active', True):
        return None
    
    # Verify password
    if verify_password(password, user_data['password_hash']):
        return user_data
    
    return None

def is_user_authenticated() -> bool:
    """Check if user is currently authenticated"""
    
    if not st.session_state.get('user_authenticated', False):
        return False
    
    # Check session token validity
    if not st.session_state.get('session_token'):
        return False
    
    # Check session timeout (24 hours)
    current_user = st.session_state.get('current_user')
    if current_user:
        login_time = current_user.get('login_time')
        if login_time and datetime.now() - login_time > timedelta(hours=24):
            logout_user()
            return False
    
    return True

def is_rate_limited() -> bool:
    """Check if user is rate limited"""
    
    attempts = st.session_state.get('login_attempts', 0)
    last_attempt = st.session_state.get('last_login_attempt')
    
    if attempts >= 5 and last_attempt:
        # Rate limit for 15 minutes
        if datetime.now() - last_attempt < timedelta(minutes=15):
            return True
        else:
            # Reset attempts after cooldown
            st.session_state.login_attempts = 0
    
    return False

def render_user_menu():
    """Render user menu in sidebar when authenticated"""
    
    if not is_user_authenticated():
        return
    
    current_user = st.session_state.get('current_user', {})
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Account")
        
        # User info
        st.markdown(f"""
        <div style="
            background: #F8FAFC;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        ">
            <div style="font-weight: 600; color: #2E5AAC;">{current_user.get('full_name', 'User')}</div>
            <div style="font-size: 0.85rem; color: #64748B;">{current_user.get('email', '')}</div>
            <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 4px;">
                Role: {current_user.get('role', 'user').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # User actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👤 Profile", use_container_width=True):
                show_user_profile()
        
        with col2:
            if st.button("🚪 Logout", use_container_width=True):
                logout_user()
                st.rerun()

def show_user_profile():
    """Show user profile modal"""
    
    current_user = st.session_state.get('current_user', {})
    
    st.subheader("👤 User Profile")
    
    # Profile information
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Full Name:**", current_user.get('full_name', 'N/A'))
        st.write("**Username:**", current_user.get('username', 'N/A'))
        st.write("**Email:**", current_user.get('email', 'N/A'))
    
    with col2:
        st.write("**Role:**", current_user.get('role', 'N/A').title())
        st.write("**Login Time:**", current_user.get('login_time', 'N/A'))
        
        # Show permissions
        permissions = current_user.get('permissions', [])
        st.write("**Permissions:**", ', '.join(permissions))
    
    # Change password section
    with st.expander("🔐 Change Password"):
        with st.form("change_password_form"):
            current_password = st.text_input(
                "Current Password",
                type="password"
            )
            
            new_password = st.text_input(
                "New Password", 
                type="password",
                help="Password must be at least 8 characters"
            )
            
            confirm_password = st.text_input(
                "Confirm New Password",
                type="password"
            )
            
            if st.form_submit_button("🔐 Change Password"):
                handle_password_change(current_password, new_password, confirm_password)

def handle_password_change(current_password: str, new_password: str, confirm_password: str):
    """Handle password change request"""
    
    # Validate inputs
    if not all([current_password, new_password, confirm_password]):
        st.error("❌ All password fields are required")
        return
    
    if new_password != confirm_password:
        st.error("❌ New passwords do not match")
        return
    
    if len(new_password) < 8:
        st.error("❌ Password must be at least 8 characters long")
        return
    
    # Verify current password
    current_user = st.session_state.get('current_user', {})
    username = current_user.get('username')
    
    if not username:
        st.error("❌ User session invalid")
        return
    
    user_data = st.session_state.user_database.get(username)
    if not user_data or not verify_password(current_password, user_data['password_hash']):
        st.error("❌ Current password is incorrect")
        return
    
    # Update password
    st.session_state.user_database[username]['password_hash'] = hash_password(new_password)
    st.success("✅ Password changed successfully!")

def logout_user():
    """Logout current user"""
    
    # Clear authentication state
    st.session_state.user_authenticated = False
    st.session_state.current_user = None
    st.session_state.session_token = None
    st.session_state.login_attempts = 0

def handle_forgot_password():
    """Handle forgot password request"""
    
    st.info("""
    🔐 **Password Recovery**
    
    For demo purposes, here are the default passwords:
    - **admin**: admin123
    - **analyst**: analyst123  
    - **viewer**: viewer123
    
    In a production environment, this would send a password reset email.
    """)

def hash_password(password: str) -> str:
    """Hash password using secure method"""
    
    # Generate salt
    salt = secrets.token_hex(32)
    
    # Hash password with salt
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    
    # Return salt + hash
    return salt + password_hash.hex()

def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash"""
    
    try:
        # Extract salt (first 64 characters)
        salt = stored_hash[:64]
        stored_password_hash = stored_hash[64:]
        
        # Hash provided password with same salt
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        
        # Compare hashes
        return hmac.compare_digest(password_hash.hex(), stored_password_hash)
    
    except Exception:
        return False

def generate_session_token(username: str) -> str:
    """Generate secure session token"""
    
    timestamp = datetime.now().isoformat()
    data = f"{username}:{timestamp}:{secrets.token_hex(16)}"
    
    return hashlib.sha256(data.encode()).hexdigest()

def check_permission(required_permission: str) -> bool:
    """Check if current user has required permission"""
    
    if not is_user_authenticated():
        return False
    
    current_user = st.session_state.get('current_user', {})
    user_permissions = current_user.get('permissions', [])
    
    return required_permission in user_permissions or 'admin' in user_permissions

def require_permission(required_permission: str):
    """Decorator-like function to require specific permission"""
    
    if not check_permission(required_permission):
        st.error(f"❌ Access denied. Required permission: {required_permission}")
        st.stop()

def render_permission_denied():
    """Render permission denied page"""
    
    st.error("🚫 Access Denied")
    st.markdown("""
    You don't have permission to access this feature.
    
    Please contact your administrator if you believe this is an error.
    """)
    
    if st.button("🔙 Go Back"):
        st.switch_page("01_🏠_Home.py")

def create_user_account(
    username: str,
    password: str,
    email: str,
    full_name: str,
    role: str = "viewer",
    permissions: List[str] = None
) -> bool:
    """Create new user account (admin only)"""
    
    # Check admin permission
    if not check_permission('admin'):
        return False
    
    # Validate inputs
    if not all([username, password, email, full_name]):
        return False
    
    # Check if username already exists
    if username in st.session_state.user_database:
        return False
    
    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False
    
    # Set default permissions based on role
    if permissions is None:
        role_permissions = {
            'admin': ['read', 'write', 'delete', 'admin'],
            'analyst': ['read', 'write'],
            'viewer': ['read']
        }
        permissions = role_permissions.get(role, ['read'])
    
    # Create user account
    st.session_state.user_database[username] = {
        'password_hash': hash_password(password),
        'role': role,
        'email': email,
        'full_name': full_name,
        'created_at': datetime.now(),
        'last_login': None,
        'permissions': permissions,
        'active': True
    }
    
    return True

def render_user_management():
    """Render user management interface (admin only)"""
    
    if not check_permission('admin'):
        render_permission_denied()
        return
    
    st.subheader("👥 User Management")
    
    # Tabs for different user management functions
    tab1, tab2 = st.tabs(["👥 User List", "➕ Add User"])
    
    with tab1:
        render_user_list()
    
    with tab2:
        render_add_user_form()

def render_user_list():
    """Render list of all users"""
    
    users = st.session_state.get('user_database', {})
    
    if not users:
        st.info("No users found")
        return
    
    # Create user data for display
    user_data = []
    for username, data in users.items():
        user_data.append({
            'Username': username,
            'Full Name': data.get('full_name', 'N/A'),
            'Email': data.get('email', 'N/A'),
            'Role': data.get('role', 'N/A').title(),
            'Status': 'Active' if data.get('active', True) else 'Inactive',
            'Last Login': data.get('last_login', 'Never')
        })
    
    # Display user table
    df = pd.DataFrame(user_data)
    st.dataframe(df, use_container_width=True)

def render_add_user_form():
    """Render form to add new user"""
    
    with st.form("add_user_form"):
        st.markdown("#### Add New User")
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username", help="Must be unique")
            full_name = st.text_input("Full Name")
            email = st.text_input("Email")
        
        with col2:
            password = st.text_input("Password", type="password", help="Minimum 8 characters")
            role = st.selectbox("Role", ["viewer", "analyst", "admin"])
            active = st.checkbox("Active", value=True)
        
        if st.form_submit_button("➕ Create User", type="primary"):
            if create_user_account(username, password, email, full_name, role):
                st.success(f"✅ User '{username}' created successfully!")
            else:
                st.error("❌ Failed to create user. Check all fields and ensure username is unique.")

# Initialize auth on import
if 'auth_initialized' not in st.session_state:
    initialize_auth_state()
