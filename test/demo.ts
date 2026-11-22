// A test app with a model to index.
import { AppConfig } from '@ghuri/core'
import { CharField, IntegerField, type Manager, Model, TextField } from '@ghuri/orm'

export class Post extends Model({
  title: new CharField({ maxLength: 200 }),
  body: new TextField({ blank: true, default: '' }),
  status: new CharField({ maxLength: 20, default: 'published' }),
  year: new IntegerField({ default: 2025 }),
}) {
  declare static objects: Manager<Post>
  static Meta = { appLabel: 'demo' }
}

export class DemoConfig extends AppConfig {
  name = 'demo'
  override label = 'demo'
  override models = { Post }
}
