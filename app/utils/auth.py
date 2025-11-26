import streamlit as st
import hashlib


class AuthManager:
    """Gestor de autenticación simple"""

    def __init__(self):
        # Usuarios permitidos
        self.users = {
            'usuario1@example.com': self._hash_password('password123'),
            'usuario2@example.com': self._hash_password('password456')
        }

        # Inicializar session state para autenticación
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_email' not in st.session_state:
            st.session_state.user_email = None

    def _hash_password(self, password: str) -> str:
        """Hash de contraseña usando SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()

    def authenticate(self, email: str, password: str) -> bool:
        """Autentica un usuario"""
        if email in self.users:
            password_hash = self._hash_password(password)
            if self.users[email] == password_hash:
                st.session_state.authenticated = True
                st.session_state.user_email = email
                return True
        return False

    def logout(self):
        """Cierra la sesión del usuario"""
        st.session_state.authenticated = False
        st.session_state.user_email = None

    def is_authenticated(self) -> bool:
        """Verifica si hay un usuario autenticado"""
        return st.session_state.get('authenticated', False)

    def get_current_user(self) -> str:
        """Obtiene el email del usuario actual"""
        return st.session_state.get('user_email', None)

    def show_login_form(self):
        """Muestra el formulario de login"""
        st.title("🔬 Clasificador de Cáncer de Mama")
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.subheader("🔐 Iniciar Sesión")

            with st.form("login_form"):
                email = st.text_input(
                    "Correo Electrónico",
                    placeholder="usuario@example.com",
                    key="login_email"
                )

                password = st.text_input(
                    "Contraseña",
                    type="password",
                    placeholder="Ingresa tu contraseña",
                    key="login_password"
                )

                submit_button = st.form_submit_button("Iniciar Sesión", use_container_width=True)

                if submit_button:
                    if not email or not password:
                        st.error("Por favor ingresa correo y contraseña")
                    else:
                        if self.authenticate(email, password):
                            st.success(f"Bienvenido/a {email}")
                            st.balloons()
                            st.experimental_rerun()
                        else:
                            st.error("Correo o contraseña incorrectos")

            # st.markdown("---")
            # with st.expander("ℹ️ Credenciales de prueba"):
            #     st.markdown("""
            #     **Usuario 1:**
            #     - Email: `usuario1@example.com`
            #     - Contraseña: `password123`

            #     **Usuario 2:**
            #     - Email: `usuario2@example.com`
            #     - Contraseña: `password456`
            #     """)

            # st.info("💡 Este es un sistema de autenticación de prueba. En producción, las credenciales deben estar en una base de datos segura.")
