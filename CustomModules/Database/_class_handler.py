

def control_transactions(Obj_SQLServer):
  
   """ Decorate SQLServer class and assert database connection before transactions and queries among other functionalities """
    
    require_connection = [
        'execute_statement'
        'query_data',
        'list_database',
        'get_servername',
        'get_database',
        'create_database',
        'set_database',
        'detail_table', 
        'select', 
        'list_tables', 
        'insert', 
        'export_to_file'
    ]
    
    class Handler(Obj_SQLServer):

        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
        
        def __str__(self):
            info_labels = ['Server:', 'Database:', 'User:', 'DSN:', 'Driver:', 'Connected:']
            info_values = [self.current_server, self.current_database, self.user, self.dsn, self.driver, self.connected]
            repr_values = [
                label +' '+ Handler.handle_db_info(info) + '\n' 
                for label, info in zip(info_labels, info_values)
            ]
            return ''.join(repr_values)

        def __getattribute__(self, name):
            attribute = object.__getattribute__(self, name)
            if not callable(attribute):
                return attribute
            else:
                if attribute.__name__ in require_connection:
                    if self.connected:
                        return attribute
                    else:
                        return Handler.connect_notification
                else:
                    return attribute

        @staticmethod
        def connect_notification(*args, **kwargs):
            print('No connection to SQL Server, use SQLServer.connect(auth)')
            return None

        @staticmethod
        def handle_db_info(db_attr_):
            if isinstance(db_attr_, bool) or db_attr_ is None:
                return str(db_attr_)
            else:
                return db_attr_        
    return Handler
